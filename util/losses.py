import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

    
def compute_intermediate_loss(config, lm_head, model_dim, lm_logits=None, labels=None, all_hidden_states=None, layer_transformation=None):
    """ train both the last and intermediate layers for Early-Exit or Shallow-Deep framework
    """
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        labels = labels.to(all_hidden_states[-1].device)
            
        if config.intermediate_loss_fn == 'ce':
            assert all_hidden_states is not None
            scale_factor = model_dim ** -0.5 if config.tie_word_embeddings else 1
            all_lm_logits = [lm_head(all_hidden_states[i] * scale_factor) for i in range(len(all_hidden_states))]
            loss = sum([loss_fct(ll.view(-1, ll.size(-1)), labels.view(-1)) for ll in all_lm_logits]) / len(all_lm_logits)

        elif config.intermediate_loss_fn == 'weighted_ce':
            assert all_hidden_states is not None
            scale_factor = model_dim ** -0.5 if config.tie_word_embeddings else 1
            all_lm_logits = [lm_head(all_hidden_states[i] * scale_factor) if (config.exit_min_layer and i >= config.exit_min_layer) \
                else lm_head(all_hidden_states[i] * scale_factor) for i in range(len(all_hidden_states))]
            loss = sum([(idx + 1) * loss_fct(ll.view(-1, ll.size(-1)), labels.view(-1)) for idx, ll in enumerate(all_lm_logits)])
            loss = loss / sum([idx + 1 for idx in range(len(all_lm_logits))])
            
        elif config.intermediate_loss_fn in ['shallowdeep_ce', 'shallowdeep_kd_dyna', 'shallowdeep_kd_last', 'shallowdeep_kd_unif']:
            assert all_hidden_states is not None
            assert config.shallow_exit_layer is not None
            trained_layers = [config.shallow_exit_layer, len(all_hidden_states)]
            scale_factor = model_dim ** -0.5 if config.tie_word_embeddings else 1
            shallowdeep_lm_logits = [lm_head(all_hidden_states[i-1] * scale_factor) for i in trained_layers]
            loss = sum([idx * loss_fct(ll.view(-1, ll.size(-1)), labels.view(-1)) for idx, ll in zip(trained_layers, shallowdeep_lm_logits)])
            loss = loss / sum(trained_layers)

            if 'shallowdeep_kd' in config.intermediate_loss_fn:
                layer_kd_loss = compute_layerwise_distillation_loss(config, all_hidden_states, layer_transformation)
                loss = loss + config.distill_layer_alpha * layer_kd_loss
            
        else:
            raise NotImplementedError
            
    return loss  


def compute_layerwise_distillation_loss(config, all_hidden_states=None, layer_transformation=None):
    """ layerwise knowledge distillation loss for the intermediate layers
    """
    TEACHER_LAYERS = {24: {12: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
                           10: [2, 5, 7, 10, 12, 15, 17, 19, 21, 23],
                           8: [2, 5, 8, 11, 14, 17, 20, 23], 
                           6: [3, 7, 11, 15, 19, 23],
                           4: [5, 11, 17, 23]},
                      12: {3: [3, 7, 11]}}

    mse_loss = torch.nn.MSELoss(reduction='mean')

    # hidden state distillation loss
    teacher_layer_output = [all_hidden_states[i].detach() for i in range(len(all_hidden_states))]
    student_layer_output = [all_hidden_states[i] for i in range(config.shallow_exit_layer)]
        
    if config.intermediate_loss_fn == 'shallowdeep_kd_last':
        # The layer mapping function is to match the last layer of
        # deep decoder model and shallow decoder model.
        if layer_transformation is not None:
            specified_student_layer_reps = layer_transformation(student_layer_output[-1])
        else:
            specified_student_layer_reps = student_layer_output[-1]
        specified_teacher_layer_reps = teacher_layer_output[-1]
        layer_kd_loss = mse_loss(specified_teacher_layer_reps, specified_student_layer_reps)
        
    elif config.intermediate_loss_fn == 'shallowdeep_kd_unif':
        # The layer mapping function is to uniformly match 
        # the deep model layer to that of shallow model layer.
        specified_teacher_layers = TEACHER_LAYERS[config.num_layers][config.shallow_exit_layer]
        
        if layer_transformation is not None:
            specified_student_layer_reps = [layer_transformation(s_layer_o) for s_layer_o in student_layer_output]
        else:
            specified_student_layer_reps = [s_layer_o for s_layer_o in student_layer_output]
        specified_teacher_layer_reps = [teacher_layer_output[i] for i in specified_teacher_layers]
        
        layer_kd_loss = 0.
        for t_layer_o, s_layer_o in zip(specified_teacher_layer_reps, specified_student_layer_reps):
            layer_kd_loss += mse_loss(t_layer_o, s_layer_o)
        layer_kd_loss /= len(specified_teacher_layer_reps)

    elif config.intermediate_loss_fn == 'shallowdeep_kd_dyna':
        # The layer mapping function is dynamically determined during the training process
        # to match a deep model layer to its closest layer in the shallow model.
        specified_teacher_layers = TEACHER_LAYERS[config.num_layers][config.shallow_exit_layer]

        if layer_transformation is not None:
            specified_student_layer_reps = [layer_transformation(s_layer_o) for s_layer_o in student_layer_output]
        else:
            specified_student_layer_reps = [s_layer_o for s_layer_o in student_layer_output]
        specified_teacher_layer_reps = [teacher_layer_output[i] for i in specified_teacher_layers]

        device = student_layer_output[0].device
        l = []
        for t_layer_o in specified_teacher_layer_reps:
            for i, s_layer_o in enumerate(specified_student_layer_reps):
                l.append(mse_loss(t_layer_o, s_layer_o))
        layerwise_loss = torch.stack(l).reshape(
            len(specified_teacher_layer_reps), len(student_layer_output)
        )

        layer_kd_loss = 0
        last_aligned_layer = len(all_hidden_states)
        alignment = []
        for search_index in range(len(specified_teacher_layers)-1, -1, -1):
            indexes = layerwise_loss[search_index].sort()[1]
            align = indexes[indexes < last_aligned_layer]

            align = align[0] if len(align) > 0 else last_aligned_layer
            alignment.append(align)
            last_aligned_layer = align

        alignment.reverse()
        alignment = torch.tensor(alignment).to(device)

        layerwise = torch.arange(len(specified_teacher_layers)).to(device)
        layer_kd_loss += layerwise_loss[layerwise, alignment].mean()
        
    else:
        raise NotImplementedError
                                
    return layer_kd_loss
      

def compute_cm_head_loss(config, lm_head, cm_head, model_dim, all_hidden_states=None):
    """ 
    train cm_head for "meta" confidence measure.
    "meta" confidence measure aims to output probability for the exit when the input is hidden_states.
    """
    scale_factor = model_dim ** -0.5 if config.tie_word_embeddings else 1
    
    if config.shallow_exit_layer is not None:
        trained_layers = [config.shallow_exit_layer, len(all_hidden_states)]
        all_lm_logits = [lm_head(all_hidden_states[i-1] * scale_factor) for i in trained_layers]
    else:
        all_lm_logits = [lm_head(all_hidden_states[i-1] * scale_factor) for i in range(len(all_hidden_states))]
    all_lm_argmax = all_lm_logits[-1].argmax(-1)
    
    device = all_lm_argmax[-1].device
    meta_labels, meta_preds = torch.empty(0).to(device), torch.empty(0).to(device)
    for idx, h in enumerate(all_hidden_states[:-1]):
        labels_ = (all_lm_logits[idx].argmax(-1) == all_lm_argmax).view(-1)  # (bsz, len) -> (bsz * len)
        meta_labels = torch.cat([meta_labels, labels_], dim=0)  # (bsz * len)
        meta_preds = torch.cat([meta_preds, cm_head(h.reshape(-1, h.size(-1)))], dim=0)  # (bsz * len, 2)

    # balanced loss
    pos, neg = sum(meta_labels) / len(meta_labels), 1 - sum(meta_labels) / len(meta_labels)
    bal_prior = torch.log(torch.tensor([neg, pos])).view(1, -1).to(device)
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(meta_preds + bal_prior, meta_labels.long())  # Logit Adjustment
    return loss