"""
T5: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L19
"""
from typing import Optional, Tuple, Union

import os
import copy
import math
import datetime
import warnings
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions, 
    Seq2SeqLMOutput
)
from transformers.models.longt5.modeling_longt5 import (
    LongT5LayerNorm,
    LongT5LayerLocalSelfAttention,
    LongT5LayerTransientGlobalSelfAttention,
    LongT5LayerFF,
    LongT5Block,
    LongT5Stack,
    LongT5ForConditionalGeneration,
    _get_local_attention_mask,
)
from transformers.models.longt5.configuration_longt5 import LongT5Config
from transformers.utils import logging

from .modeling_t5 import EffT5LayerSelfAttention as LongT5LayerSelfAttention
from .modeling_t5 import EffT5LayerCrossAttention as LongT5LayerCrossAttention
from util import (
    compute_intermediate_loss,
    compute_cm_head_loss,
    split_tensors_by_mask, 
    restore_tensors_by_mask,
    get_skip_mask,
)

logger = logging.get_logger(__name__)
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class EffLongT5Block(LongT5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.is_decoder = config.is_decoder
        if config.is_decoder:
            attention_layer = LongT5LayerSelfAttention
        elif config.encoder_attention_type == "local":
            attention_layer = LongT5LayerLocalSelfAttention
        elif config.encoder_attention_type == "transient-global":
            attention_layer = LongT5LayerTransientGlobalSelfAttention
        else:
            raise ValueError(
                "For encoder attention mechanism, either `local` or `transient-global` attention type is expected, "
                f"but got {config.encoder_attention_type}."
            )
        
        self.layer = nn.ModuleList()
        self.layer.append(attention_layer(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(LongT5LayerCrossAttention(config))

        self.layer.append(LongT5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        skip_mask=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            skip_mask=skip_mask,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=skip_mask,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        if skip_mask is None:
            hidden_states = self.layer[-1](hidden_states)
        else:
            if skip_mask.sum().item() != hidden_states.shape[0]:
                keep_hidden_states, hidden_states, ids_restore = split_tensors_by_mask(hidden_states, skip_mask)
                keep_hidden_states = self.layer[-1](keep_hidden_states)
                hidden_states = restore_tensors_by_mask(keep_hidden_states, hidden_states, ids_restore)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class EffLongT5Stack(LongT5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1
        
        self.block = nn.ModuleList(
            [EffLongT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Early-Exit framework
        self.use_early_exit = False
        if self.is_decoder and config.exit_conf_type is not None:
            self.use_early_exit = True
            
        # Shallow-Deep framework
        self.use_shallow_deep = config.use_shallow_deep
        self.shallow_exit_layer = config.shallow_exit_layer
        if self.is_decoder and config.use_shallow_deep:
            assert config.shallow_exit_layer > 0 and config.shallow_exit_layer < len(self.block)

        self.block_op = [0] * config.num_layers  # to calculate the average number of forward block layers
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        lm_head=None,
        cm_head=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # We use local attention in encoder self-attention, otherwise standard self & cross attentions are used
        if self.is_decoder:
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        elif self.config.encoder_attention_type == "local":
            extended_attention_mask = _get_local_attention_mask(attention_mask, self.block_len, inputs_embeds.device)
        else:
            extended_attention_mask = attention_mask

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        skip_mask, self.skip_mask_cache = None, None
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if self.is_decoder and self.config.static_exit_layer is not None:
                if i == self.config.static_exit_layer: break

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                if self.is_decoder or i > 0:
                    all_hidden_states = all_hidden_states + (self.dropout(self.final_layer_norm(hidden_states)),)
                else:
                    all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                # check that tokens are generated once in a time
                auto_reg = True if hidden_states.shape[1] == 1 else False
                if self.is_decoder and (self.use_early_exit or self.use_shallow_deep) and auto_reg and i == 0: 
                    self.block_op[i] += hidden_states.shape[0]
                
                # Shallow-Deep Framework
                if self.is_decoder and auto_reg and self.use_shallow_deep and i == self.shallow_exit_layer:
                    hidden_ = copy.deepcopy(hidden_states)

                    hidden_ = self.dropout(self.final_layer_norm(hidden_))
                    logits = lm_head(hidden_) if not self.config.tie_word_embeddings \
                        else lm_head(hidden_ * (self.config.d_model ** -0.5))

                    skip_mask = get_skip_mask(
                        logits,
                        hidden_,
                        cm_head,
                        config=self.config,
                        pos_time=past_key_value[0].shape[2] + 1 if past_key_value is not None else 1,
                    )
                    
                    self.block_op[i] += (skip_mask.shape[0] - skip_mask.sum().item())
                
                # exploit early-exit mask: softmax and meta
                elif self.is_decoder and auto_reg and self.use_early_exit and i > 0:
                    if self.skip_mask_cache is None or (self.skip_mask_cache.shape[0] != self.skip_mask_cache.sum().item()):
                        hidden_ = copy.deepcopy(hidden_states)
                        if self.skip_mask_cache is not None:  # self.skip_mask_cache.shape[0] != self.skip_mask_cache.sum().item()
                            hidden_, _, ids_restore = split_tensors_by_mask(hidden_, self.skip_mask_cache)
                            _, skip_cache, _ = split_tensors_by_mask(self.skip_mask_cache, self.skip_mask_cache, ids_restore)

                        hidden_ = self.dropout(self.final_layer_norm(hidden_))
                        logits = lm_head(hidden_) if not self.config.tie_word_embeddings \
                            else lm_head(hidden_ * (self.config.d_model ** -0.5))

                        skip_mask = get_skip_mask(
                            logits,
                            hidden_,
                            cm_head,
                            config=self.config,
                            pos_time=past_key_value[0].shape[2] + 1 if past_key_value is not None else 1,
                        )
                        self.block_op[i] += (skip_mask.shape[0] - skip_mask.sum().item())

                        if self.skip_mask_cache is None:
                            self.skip_mask_cache = skip_mask
                        else:
                            skip_mask = self.skip_mask_cache = restore_tensors_by_mask(skip_mask, skip_cache, ids_restore)
                        
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    skip_mask=skip_mask,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class EffLongT5ForConditionalGeneration(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.static_exit_layer = None
        self.encoder = EffLongT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = EffLongT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        
        if self.config.exit_conf_type == 'meta' or self.config.shallow2deep_conf_type:
            self.cm_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model, bias=False),
                nn.ReLU(),
                nn.Linear(config.d_model, 2, bias=False),
            )
        else: self.cm_head = None
        
        
        if self.config.intermediate_loss_fn is not None and ('shallowdeep_kd' in self.config.intermediate_loss_fn) and self.config.do_layer_transformation:
            self.layer_transformation = nn.Linear(
                config.hidden_size, config.hidden_size)
        else: self.layer_transformation = None
        
        self.deploy_time = None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        EffT5ForConditionalGeneration class is for the evaluation of LLMs.
        This class supports the batch size larger than 1, via two functions:

        1) split_tensors_by_mask : split tensors with skip_mask, which decides to keep or skip the forward path.
        2) restore_tensors_by_mask : restore splited tensors to the original order by ids_restore.

        eval_time or avg_num_blocks, measured by this class, is not correct to our intention of methodology
        since skipped tokens just sequentially copy hidden_states and they should wait until other samples in the batch are finished.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        output_hidden_states = True
        if not self.config.train_meta_cm_head:
            encoder_outputs, decoder_outputs = self.forward_impl(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                                                                    head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs,
                                                                    past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache,
                                                                    output_attentions, output_hidden_states, return_dict)
        else:
            # for training meta cm_head
            with torch.no_grad():
                encoder_outputs, decoder_outputs = self.forward_impl(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                                                                    head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs,
                                                                    past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache,
                                                                    output_attentions, output_hidden_states, return_dict)
        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
        
        
        lm_logits = self.lm_head(sequence_output)

        if not self.config.output_hidden_states_decoder or not self.training:
            loss = self.compute_model_loss(lm_logits, labels)
        elif self.config.intermediate_loss_fn is not None:
            loss = compute_intermediate_loss(self.config, self.lm_head, self.model_dim, lm_logits, labels, decoder_outputs[2][1:], self.layer_transformation)
        elif self.config.train_meta_cm_head:
            loss = compute_cm_head_loss(self.config, self.lm_head, self.cm_head, self.model_dim, decoder_outputs[2][1:])

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def compute_model_loss(self, lm_logits=None, labels=None):
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            assert lm_logits is not None
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        return loss
    
    def forward_impl(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=self.config.output_hidden_states_decoder,
            return_dict=return_dict,
            lm_head=self.lm_head,
            cm_head=self.cm_head,
        )
        return encoder_outputs, decoder_outputs
