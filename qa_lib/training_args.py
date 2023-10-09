from transformers.training_args import OptimizerNames


def adjust_training_args(training_args, data_args, additional_args):

    # train
    training_args.optim = OptimizerNames.ADAFACTOR

    # save ckpt
    training_args.save_total_limit = 1
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = 'f1' if 'squad' in data_args.dataset_name else 'rougeLsum'
    training_args.greater_is_better = True
    training_args.evaluation_strategy = 'steps'
    training_args.eval_steps = training_args.save_steps
    training_args.intermediate_loss_fn = additional_args.intermediate_loss_fn

    if training_args.do_train:
        # static_exit_layer argument should be defined only for evaluation
        assert additional_args.static_exit_layer is None
    
    if additional_args.deploy_scenario:
        # make sure to use batch size of 1 and one GPU
        assert training_args.per_device_eval_batch_size == 1 and training_args.n_gpu == 1

    return training_args