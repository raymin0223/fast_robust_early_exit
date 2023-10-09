from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AdditionalArguments:
    """
    Arguments for accelerating decoder models.
    """

    # deployment scenario
    deploy_scenario: Optional[bool] = field(
        default=False, metadata={"help": ("Assume a deploying scneario for the accurate measurement.")},
    )
    use_synchronize: Optional[bool] = field(
        default=True, metadata={"help": ("Use synchronize when measuring the inference time.")},
    )

    # train the intermediate layers as well
    output_hidden_states_decoder: Optional[bool] = field(
        default=False, metadata={"help": ("Output all hidden states in decoder model to train intermedidate layers.")},
    )
    intermediate_loss_fn: Optional[str] = field(
        default=None, metadata={"help": ("Choose the loss function to train intermediate layers as well.")},
    )
    distill_layer_alpha: Optional[float] = field(
        default=None, metadata={"help": ("Distillation interpolation hyperparameter between CrossEntropy and KL divergence.")}
    )
    do_layer_transformation: Optional[bool] = field(
        default=False, metadata={"help": ("Whether or not use transformation for student (shallow decoder) hidden states")}
    )

    # static: output all tokens after a specific layer, not the end of decodoer layer
    static_exit_layer: Optional[int] = field(
        default=None, metadata={"help": ("Choose an exit block for all tokens (i.e., exit tokens after [static_exit_layer] block).")},
    )

    # early-exit: output tokens based on confidence in decoder layers
    use_early_exit: Optional[bool] = field(
        default=False, metadata={"help": ("Use early-exit framework in decoder model.")}
    )
    exit_conf_type: Optional[str] = field(
        default=None, metadata={"help": ("Select the type of confidence measure.")},
    )   
    exit_conf_threshold: Optional[float] = field(
        default=1., metadata={"help": ("Default threshold value for early-exit framework.")},
    )
    exit_position_temp: Optional[float] = field(
        default=None, metadata={"help": ("Temperature value for decaying confidence threshold")},
    )
    exit_min_layer: Optional[int] = field(
        default=None, metadata={"help": ("To address unstable text generation and training, exit after certain layers.")},
    )   
    train_meta_cm_head: Optional[bool] = field(
        default=False, metadata={"help": ("Train cm (confidence measure) head to align last hidden_states when exit_conf_type is set to meta.")}
    )
        
    # shallow-deep framework
    use_shallow_deep: Optional[bool] = field(
        default=False, metadata={"help": ("Use shallow-deep decoder framework in decoder model.")}
    )
    shallow_exit_layer: Optional[int] = field(
        default=None, metadata={"help": ("Number of layers for shallow decoder model.")}
    )
    shallow2deep_conf_type: Optional[str] = field(
        default=None, metadata={"help": ("Select the type of confidence measure for chaning shallow to deep decoder.")},
    )   
    shallow2deep_conf_threshold: Optional[float] = field(
        default=1., metadata={"help": ("Default threshold value in Shallow-Deep framework.")},
    )
    parallel_gen_token: Optional[bool] = field(
        default=True, metadata={"help": ("With the previous skipped tokens, generate the next token of Deep decoder in a non-autoregressive manner.")},
    )
    copy_skipped_hidden_states: Optional[bool] = field(
        default=False, metadata={"help": ("For the previous skipped tokens, copy hidden_states and generate key_value of Deep decoder.")},
    )
    parallel_causal_mask: Optional[bool] = field(
        default=True, metadata={"help": ("Using causal masking for sequence parallel computing when shallow2deep occurs.")}
    )
    rollback_conf_threshold: Optional[float] = field(
        default=None, metadata={"help": ("Default threshold value for RollBack policy in Shallow-Deep framework.")},
    )
        
    # adpative threshold estimator
    use_adapt_threshold: Optional[bool] = field(
        default=False, metadata={"help": ("Using adaptive threshold estimator for FREE framework.")},
    )
    
    # low rank adaptation
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": ("Using low-rank adaptation for large language models")}
    )
    lora_rank: Optional[int] = field(
        default=8, metadata={"help": ("Default rank value of lora")},
    )
    lora_alpha: Optional[float] = field(
        default=8, metadata={"help": ("Default alpha value of lora")},
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": ("Default dropout value of lora")}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None, metadata={"help": ("Change target modules of lora")}
    )
    

def update_autoconfig(config, additional_args, **kwargs):

    # assertion
    if additional_args.intermediate_loss_fn is not None:
        assert additional_args.output_hidden_states_decoder
    if additional_args.train_meta_cm_head:
        assert additional_args.output_hidden_states_decoder
        assert additional_args.intermediate_loss_fn is None  # when training cm_head, model should be fully fine-tuned
    if additional_args.use_shallow_deep:
        assert additional_args.shallow_exit_layer is not None
    if not additional_args.parallel_causal_mask:
        assert not additional_args.copy_skipped_hidden_states
        assert additional_args.rollback_conf_threshold is None
    if additional_args.rollback_conf_threshold is not None:
        assert not additional_args.copy_skipped_hidden_states

    deploy_config = {
        'use_synchronize': additional_args.use_synchronize,
    }
    config.update(deploy_config)
    
    inter_config = {
        'output_hidden_states_decoder': additional_args.output_hidden_states_decoder,
        'intermediate_loss_fn': additional_args.intermediate_loss_fn,
        'distill_layer_alpha': additional_args.distill_layer_alpha,
        'do_layer_transformation': additional_args.do_layer_transformation,
    }
    config.update(inter_config)

    static_config = {
        'static_exit_layer': additional_args.static_exit_layer,
    }
    config.update(static_config)
    
    early_exit_config = {
        'use_early_exit': additional_args.use_early_exit,
        'exit_conf_type': additional_args.exit_conf_type,
        'exit_conf_threshold': additional_args.exit_conf_threshold,
        'exit_position_temp': additional_args.exit_position_temp,
        'exit_min_layer': additional_args.exit_min_layer,
        'train_meta_cm_head': additional_args.train_meta_cm_head,
        'max_answer_length': kwargs.get('max_answer_length', None),
    }
    config.update(early_exit_config)
    
    shallow_deep_config = {
        'use_shallow_deep': additional_args.use_shallow_deep,
        'shallow_exit_layer': additional_args.shallow_exit_layer,
        'shallow2deep_conf_type': additional_args.shallow2deep_conf_type,
        'shallow2deep_conf_threshold': additional_args.shallow2deep_conf_threshold,  
        'parallel_gen_token': additional_args.parallel_gen_token,  
        'copy_skipped_hidden_states': additional_args.copy_skipped_hidden_states,  
        'rollback_conf_threshold': additional_args.rollback_conf_threshold,
        'parallel_causal_mask': additional_args.parallel_causal_mask
    }
    config.update(shallow_deep_config)
    
    adaptive_threshold_config = {
        "use_adapt_threshold": additional_args.use_adapt_threshold,
    }
    config.update(adaptive_threshold_config)
    
    lora_config = {
        "use_lora": additional_args.use_lora,
        "lora_rank": additional_args.lora_rank,
        "lora_alpha": additional_args.lora_alpha,
        "lora_dropout": additional_args.lora_dropout,
        "lora_target_modules": additional_args.lora_target_modules,
    }
    config.update(lora_config)

    return config
