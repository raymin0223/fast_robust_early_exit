"""
T5: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L19
"""
from typing import Optional, Tuple, Union, List, Callable

import os
import copy
import math
import time
import datetime
import warnings
import numpy as np
import torch
import torch.distributed as dist
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
from transformers.generation.utils import GreedySearchDecoderOnlyOutput, GreedySearchEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList, validate_stopping_criteria
from transformers.utils import logging

from .deploying_t5 import DeployT5LayerSelfAttention
from .deploying_t5 import DeployT5LayerCrossAttention
from util import (
    get_skip_mask,
    BetaMixture1D,
)

logger = logging.get_logger(__name__)
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]

class DeployLongT5Block(LongT5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.is_decoder = config.is_decoder
        if config.is_decoder:
            attention_layer = DeployT5LayerSelfAttention
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
            self.layer.append(DeployT5LayerCrossAttention(config))

        self.layer.append(LongT5LayerFF(config))

    def gen_cross_attn_key_value(
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
    ):
        r""" 
        In Shallow-Deep framework, if all previous tokens, including <start> token, have skipped Deep decoder,
        generate cross-attn key_values only ONCE because they are shared for all sequence.
        
        return (None, None) + cross_attn_past_key_value: Tuple[torch.Tensor] (length of 2)
        """

        # if all previous tokens, including <start> token, have skipped Deep decoder
        assert self.is_decoder and encoder_hidden_states is not None
        cross_attn_past_key_value = self.layer[1](
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            gen_cross_attn_key_value=True,
        )
        self.key_value_gen_time = self.layer[1].key_value_gen_time

        past_key_value = [None, None,] + cross_attn_past_key_value
        return past_key_value

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
        skip_mask=False,
        parallel_mask=False,
        stack_hidden_states=None,
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
            stack_hidden_states=stack_hidden_states,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights
        
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
                parallel_mask=parallel_mask,
            )
            hidden_states = cross_attention_outputs[0]

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        # Apply Feed Forward layer
        if not skip_mask:
            hidden_states = self.layer[-1](hidden_states)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder:
            self.ffn_time = datetime.datetime.now() - start
            self.key_value_gen_time = (self.layer[0].key_value_gen_time, self.layer[1].key_value_gen_time)
            self.attn_time = (self.layer[0].attn_ffn_time, self.layer[1].attn_ffn_time)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class DeployLongT5Stack(LongT5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        
        self.local_radius = config.local_radius
        self.block_len = self.local_radius + 1

        self.block = nn.ModuleList(
            [DeployLongT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.device_map = None
        self.gradient_checkpointing = False

        # Early-Exit framework
        self.use_early_exit = config.use_early_exit
        self.exit_min_layer = config.exit_min_layer
            
        # Shallow-Deep framework
        self.use_shallow_deep = config.use_shallow_deep
        self.shallow_exit_layer = config.shallow_exit_layer
        if self.is_decoder and config.use_shallow_deep:
            assert config.shallow_exit_layer > 0 and config.shallow_exit_layer < len(self.block)
        
        # Synchronized Parallel Decoding
        self.block_op = [0] * config.num_layers  # to calculate the average number of forward block layers
        self.parallel_tokens_shallow = 0  # how much tokens are used in parallel decoding as stack_hidden_states
        self.parallel_tokens_deep = 0  # how much tokens are used in parallel decoding with skip_mask = False
        self.stack_hidden_states = ()  # store hidden_states that do not forward Deep decoder
        
        # Adaptive Threshold Estimator
        self.bmm_model = BetaMixture1D()
        self.bmm_threshold = None
        self.stack_conf, self.stack_pred = (), ()
        self.stack_conf_all, self.stack_ident_all = (), ()
        
        if self.is_decoder:
            self._reset_time_measure()
        else: self.deploy_time = None
        
    def _reset_time_measure(self):
        self.deploy_time = {'time_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
                            'time_attn': [datetime.timedelta(), datetime.timedelta()],
                            'time_ffn': datetime.timedelta(),
                            'time_confidence': datetime.timedelta(),
                            'time_exit_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
                            'time_exit_attn': [datetime.timedelta(), datetime.timedelta()],
                            'time_exit_ffn': datetime.timedelta(),
                            'time_parallel_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
                            'time_parallel_attn': [datetime.timedelta(), datetime.timedelta()],
                            'time_parallel_ffn': datetime.timedelta(),
                            'time_others': datetime.timedelta(),}

    def parallel_gen_token(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_extended_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        present_key_value_states=None,
        use_cache=None,
        output_attentions=None,
        layer_idx=None,
    ):
        r""" 
        if pask_key_values is not defined, it implies that all previous tokens have skipped Deep decoder.
            Because all sequences share key_value of cross-attn layer,
            we need to generate key_value of cross-attn layer only once for <start> token.
        else:
            key_values of cross-attn are already stored in 'past_key_values'.

        Then, generate the next token in a non-autoregressive manner.
        if copy_skipped_hidden_states is True,
            copy previous skipped hidden_states for Deep decoder blocks.
        else:
            attention calculate for stack_hidden_states as well.
            thus, we can utilize them in RollBack policy.
        """
        
        if not self.config.copy_skipped_hidden_states:
            hidden_states = torch.cat(self.stack_hidden_states + (hidden_states,), dim=1)            
            # reset and re-calculate based on the length of hidden_states
            extended_attention_mask, position_bias = None, None
        else:
            self.stack_hidden_states = torch.cat(self.stack_hidden_states, dim=1)
            extended_attention_mask = attention_mask

        for j in range(layer_idx, len(self.block)):
        
            past_key_value = past_key_values[j]
            if past_key_value is None:
                # if pask_key_values is not defined, it implies that all previous tokens have skipped Deep decoder
                # need to generate key_value of cross-attn layer only once for <start> token
                past_key_value = self.block[j].gen_cross_attn_key_value(
                    hidden_states,  # dummy
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=head_mask[j],
                    cross_attn_layer_head_mask=cross_attn_head_mask[j],
                    past_key_value=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                self.deploy_time['time_parallel_key_value_gen'][1] += self.block[j].key_value_gen_time
            
            if self.config.use_synchronize: torch.cuda.synchronize()
            start = datetime.datetime.now()
            if extended_attention_mask is None or position_bias is None:
                real_seq_length = hidden_states.shape[1]
                if past_key_value[0] is not None: real_seq_length += past_key_value[0].shape[2]
                key_length = real_seq_length
                
                if self.config.parallel_causal_mask and extended_attention_mask is None:
                    attention_mask = torch.ones(hidden_states.shape[0], real_seq_length, device=hidden_states.device)
                    extended_attention_mask = self.get_extended_attention_mask(attention_mask, torch.Size([hidden_states.shape[0], hidden_states.shape[1]]))
                
                if position_bias is None:      
                    position_bias = self.block[0].layer[0].SelfAttention.compute_bias(real_seq_length, key_length, device=hidden_states.device)

                    # if key and values are already calculated
                    # we want only the last query position bias
                    if past_key_value is not None:
                        position_bias = position_bias[:, :, -hidden_states.size(1):, :]

                    if extended_attention_mask is not None:
                        position_bias = position_bias + extended_attention_mask  # (batch_size, n_heads, seq_length, key_length)

            if self.config.use_synchronize: torch.cuda.synchronize()
            self.deploy_time['time_others'] += (datetime.datetime.now() - start)

            layer_outputs = self.block[j](
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=head_mask[j],
                cross_attn_layer_head_mask=cross_attn_head_mask[j],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                skip_mask=False,
                parallel_mask=True,
                stack_hidden_states=self.stack_hidden_states if self.config.copy_skipped_hidden_states else None,
            )
            for idx, t in enumerate(self.block[j].key_value_gen_time): self.deploy_time['time_parallel_key_value_gen'][idx] += t
            for idx, t in enumerate(self.block[j].attn_time): self.deploy_time['time_parallel_attn'][idx] += t
            self.deploy_time['time_parallel_ffn'] += self.block[j].ffn_time
            
            
            if self.config.use_synchronize: torch.cuda.synchronize()
            start = datetime.datetime.now()
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
                present_key_value_states = present_key_value_states + [present_key_value_state,]

            if self.config.use_synchronize: torch.cuda.synchronize()
            self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        self.stack_hidden_states = ()
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        
        return hidden_states, present_key_value_states

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
        r""" 
        We have implemented the following inference strategy:

        1) Normal framework: Forward all transformer layers.
        2) Static framework: Only forward the pre-defined number of early layers.
        3) Early-Exit framework: Each token can exit the forward path if confidence is higher than threshold.
        4) Shallow-Deep framework: 
            While a few early layers are defined as 'Shallow' decoder, the entire network including Shallow is defined as 'Deep' decoder.
            Each token can skip the Deep decoder path if confidence at Shallow decoder is higher than threshold.
        """

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
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
            self.stack_hidden_states = ()
            self.stack_conf, self.stack_pred = (), ()

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
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
        present_key_value_states = [] if use_cache else None
        all_hidden_states = None
        all_attentions = None
        all_cross_attentions = None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)

        skip_mask = False  # False: forward, and True: skip
        self.shallow2deep = False  # False: skip, and True: forward
        self.lm_logits = None  # to prevent calculating logits twice

        for i, layer_module in enumerate(self.block):
                
            # Static framework
            if self.is_decoder and self.config.static_exit_layer is not None:
                if i == self.config.static_exit_layer: break

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            
            # check that tokens are generated once in a time
            auto_reg = True if hidden_states.shape[1] == 1 else False
            if self.is_decoder and auto_reg and i == 0: self.block_op[i] += 1
                            
            if self.is_decoder and auto_reg and i > 0:
                
                # Shallow-Deep framework 
                if self.use_shallow_deep and i == self.shallow_exit_layer:
                    if self.config.use_synchronize: torch.cuda.synchronize()
                    start = datetime.datetime.now()
                    _hidden_states = self.dropout(self.final_layer_norm(hidden_states))
                    lm_logits = lm_head(_hidden_states) if not self.config.tie_word_embeddings \
                        else lm_head(_hidden_states * (self.config.d_model ** -0.5))
                        
                    skip_mask, conf = get_skip_mask(
                        lm_logits,
                        _hidden_states,
                        cm_head,
                        config=self.config,
                        adapt_threshold=self.bmm_threshold,
                        return_conf=True,
                    )
                    self.stack_conf = self.stack_conf + (conf,)
                    self.stack_pred = self.stack_pred + (lm_logits,)

                    if not skip_mask: self.block_op[i] += 1
                    if self.config.use_synchronize: torch.cuda.synchronize()
                    self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)

                    # if skip Deep decoder, store hidden_states at self.shallow_exit_layer
                    if skip_mask:
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        start = datetime.datetime.now()
                        self.lm_logits = lm_logits
                        if self.config.parallel_gen_token:
                            if use_cache:
                                for j in range(i, len(self.block)):
                                    present_key_value_states = present_key_value_states + [past_key_values[j],]
                            self.stack_hidden_states = self.stack_hidden_states + (hidden_states,)
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)
                        break

                    if not skip_mask:
                        self.shallow2deep = True
                        if self.config.parallel_gen_token and len(self.stack_hidden_states):
                            self.parallel_tokens_shallow += len(self.stack_hidden_states)
                            self.parallel_tokens_deep += 1
                            
                            # in Shallow-Deep decoder, generate the next token in a non-autoregressive manner
                            hidden_states, present_key_value_states = self.parallel_gen_token(
                                hidden_states,
                                attention_mask=extended_attention_mask,
                                position_bias=position_bias,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_extended_attention_mask=encoder_extended_attention_mask,
                                encoder_decoder_position_bias=encoder_decoder_position_bias,
                                head_mask=head_mask,
                                cross_attn_head_mask=cross_attn_head_mask,
                                past_key_values=past_key_values,
                                present_key_value_states=present_key_value_states,
                                use_cache=use_cache,
                                output_attentions=output_attentions,
                                layer_idx=self.shallow_exit_layer,
                            )
                            
                            # Adaptive Threshold Estimator
                            if self.config.use_adapt_threshold:
                                # Calibration Set Update
                                self.lm_logits = self.lm_head(self.dropout(self.final_layer_norm(hidden_states)))
                                deep_pred = self.lm_logits.argmax(-1)
                                shallow_pred = torch.cat(self.stack_pred[-deep_pred.size(1):]).argmax(-1).view(-1)

                                self.stack_conf_all += self.stack_conf[-deep_pred.size(1):]
                                self.stack_ident_all += ((deep_pred.view(-1) == shallow_pred.view(-1)).long().cpu().numpy(),)
                                self.stack_conf, self.stack_pred = (), ()
                                
                            break

                # Early-Exit framework
                elif self.use_early_exit and not skip_mask:
                    if self.exit_min_layer is not None and i < self.exit_min_layer: 
                        self.block_op[i] += 1
                    else:
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        start = datetime.datetime.now()
                        _hidden_states = self.dropout(self.final_layer_norm(hidden_states))
                        lm_logits = lm_head(_hidden_states) if not self.config.tie_word_embeddings \
                            else lm_head(_hidden_states * (self.config.d_model ** -0.5))
                            
                        skip_mask = get_skip_mask(
                            lm_logits,
                            _hidden_states,
                            cm_head,
                            config=self.config,
                            pos_time=past_key_values[i][0].shape[2] + 1 if past_key_values[i] is not None else 1
                        )
                        if not skip_mask: self.block_op[i] += 1                    
                        if skip_mask: self.lm_logits = lm_logits
                        if self.config.use_synchronize: torch.cuda.synchronize()
                        self.deploy_time['time_confidence'] += (datetime.datetime.now() - start)
                    
                # Normal framework
                elif (not self.use_shallow_deep and not self.use_early_exit):
                    self.block_op[i] += 1
                
            past_key_value = past_key_values[i]
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

            if self.is_decoder:
                if self.config.use_early_exit: prefix = 'time_exit_' if skip_mask else 'time_'
                elif self.config.use_shallow_deep: prefix = 'time_parallel_' if self.shallow2deep else 'time_'
                else: prefix = 'time_'
                for idx, t in enumerate(layer_module.key_value_gen_time): self.deploy_time[prefix + 'key_value_gen'][idx] += t
                for idx, t in enumerate(layer_module.attn_time): self.deploy_time[prefix + 'attn'][idx] += t
                self.deploy_time[prefix + 'ffn'] += layer_module.ffn_time

            if self.config.use_synchronize: torch.cuda.synchronize()
            start = datetime.datetime.now()
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
                present_key_value_states = present_key_value_states + [present_key_value_state,]
            if self.config.use_synchronize: torch.cuda.synchronize()
            if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if not skip_mask and self.lm_logits is None:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
        if self.config.use_synchronize: torch.cuda.synchronize()
        if self.is_decoder: self.deploy_time['time_others'] += (datetime.datetime.now() - start)

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


class DeployLongT5ForConditionalGeneration(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        encoder_config.static_exit_layer = None
        self.encoder = DeployLongT5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = DeployLongT5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.decoder.lm_head = self.lm_head
        if self.config.exit_conf_type == 'meta' or self.config.shallow2deep_conf_type == "meta":
            self.cm_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model, bias=True),
                nn.ReLU(),
                nn.Linear(config.d_model, 2, bias=True),
            )
        else:
            self.cm_head = None

        # RollBack policy
        self.rollback_num = 0
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        # BMM
        self.bmm_update_iter = 0
        self.bmm_update_max_iter = 100

        self.deploy_time = {
            'time_encoder_forward': datetime.timedelta(),
            'time_decoder_forward': datetime.timedelta(),
            'time_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
            'time_attn': [datetime.timedelta(), datetime.timedelta()],
            'time_ffn': datetime.timedelta(),
            'time_confidence': datetime.timedelta(),
            'time_exit_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
            'time_exit_attn': [datetime.timedelta(), datetime.timedelta()],
            'time_exit_ffn': datetime.timedelta(),
            'time_parallel_key_value_gen': [datetime.timedelta(), datetime.timedelta()],
            'time_parallel_attn': [datetime.timedelta(), datetime.timedelta()],
            'time_parallel_ffn': datetime.timedelta(),
            'time_others': datetime.timedelta(),
        }

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
        DeployT5ForConditionalGeneration class is for a deployment scenario,
        where the decoder models are communicating with only one user (i.e., the batch size of 1).

        Here, for the faster inference, we have implemented non-autoregressive hidden_state copying in Shallow-Deep framework.
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        encoder_outputs, decoder_outputs = self.forward_impl(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                                                            head_mask, decoder_head_mask, cross_attn_head_mask, encoder_outputs,
                                                            past_key_values, inputs_embeds, decoder_inputs_embeds, labels, use_cache,
                                                            output_attentions, output_hidden_states, return_dict)
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if self.decoder.lm_logits is None:  # token has not skipped
            sequence_output = decoder_outputs[0]

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)
            
            lm_logits = self.lm_head(sequence_output)
        else: lm_logits = self.decoder.lm_logits
        
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_others'] += (datetime.datetime.now() - start)
        self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)
        
        if self.config.rollback_conf_threshold is None:
            lm_logits = lm_logits[:, [-1], :]
        loss = self.compute_model_loss(lm_logits, labels)

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
        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
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
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_encoder_forward'] += (datetime.datetime.now() - start)
        
        hidden_states = encoder_outputs[0]

        if self.config.use_synchronize: torch.cuda.synchronize()
        start = datetime.datetime.now()
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
            
        if past_key_values is None and len(self.decoder.stack_conf_all) > 0 and self.bmm_update_iter < self.bmm_update_max_iter:
            X = np.hstack(self.decoder.stack_conf_all)
            Y = np.hstack(self.decoder.stack_ident_all)
            self.decoder.bmm_model.fit(X, Y)
            
            self.decoder.bmm_threshold = self.decoder.bmm_model.predict_proba(0.3, 0.9)
            self.bmm_update_iter += 1
            
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
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            lm_head=self.lm_head,
            cm_head=self.cm_head,
        )
        if self.config.use_synchronize: torch.cuda.synchronize()
        self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)
        for k, v in self.decoder.deploy_time.items():
            if type(v) != list: self.deploy_time[k] += v
            else: self.deploy_time[k] = [_d + _v for _d, _v in zip(self.deploy_time[k], v)]
        self.decoder._reset_time_measure()
        
        return encoder_outputs, decoder_outputs

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).
        """

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only

        # for RollBack policy
        self.rollback_candidates = ()
        self.pass_length_rollback = 0

        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )       

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need
            
            # RollBack policy
            if self.config.use_shallow_deep and self.decoder.shallow2deep and not self.config.copy_skipped_hidden_states and self.config.rollback_conf_threshold is not None:
                if self.config.use_synchronize: torch.cuda.synchronize()
                start = datetime.datetime.now()

                seq_len = outputs.logits.size(1)
                if seq_len == 1:
                    # stack_hidden_states is empty, so do not need to RollBack
                    assert len(self.rollback_candidates) == 0
                    self.pass_length_rollback += 1
                
                else:
                    # we should check RollBack
                    assert seq_len - 1 == len(self.rollback_candidates)
                    
                    deep_logits = outputs.logits[:, :-1, :]
                    shallow_preds = torch.cat(self.rollback_candidates, dim=0)
                    rollback_loss = self.criterion(deep_logits.squeeze(0), shallow_preds)

                    for j, _loss in enumerate(rollback_loss):
                        if _loss.item() > self.config.rollback_conf_threshold:
                            # RollBack
                            outputs.logits = deep_logits[:, [j], :]
                            
                            # remove RollBacked tokens
                            input_ids = input_ids[:, :self.pass_length_rollback + 1]  # consider sos token
                            past_key_values = []
                            for past in outputs.past_key_values:
                                past_key_values += [[past[0][:, :, :self.pass_length_rollback + 1, :],  # self-attn key
                                                     past[1][:, :, :self.pass_length_rollback + 1, :],  # self-attn value
                                                     past[2],
                                                     past[3]],]
                            outputs.past_key_values = past_key_values

                            self.decoder.block_op[0] -= (seq_len - 1) - j
                            self.rollback_num += (seq_len - 1) - j
                            break
                        else:
                            self.pass_length_rollback += 1
                    
                    self.rollback_candidates = ()
                    self.pass_length_rollback += 1
                    
                if self.config.use_synchronize: torch.cuda.synchronize()
                self.deploy_time['time_decoder_forward'] += (datetime.datetime.now() - start)

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # for RollBack, store Shallow decoder's predictions
            if self.config.use_shallow_deep and not self.decoder.shallow2deep and not self.config.copy_skipped_hidden_states and self.config.rollback_conf_threshold is not None:
                self.rollback_candidates += (next_tokens,)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
