# Fast and Robust Early-Exiting (EMNLP 2023)

<a href="https://arxiv.org/abs/2310.05424"><img src="https://img.shields.io/badge/Paper-arXiv:2310.05424-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a>

<p align="center">
<img width="1394" src="https://github.com/raymin0223/fast_robust_early_exit/assets/50742281/0aba3284-951c-4342-af1f-16dc70030654">
</p>

[**Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding**](https://arxiv.org/abs/2310.05424)       
[Sangmin Bae](https://www.raymin0223.com)$^\*$,
[Jongwoo Ko](https://sites.google.com/view/jongwooko)$^\*$,
[Hwanjun Song](https://songhwanjun.github.io)$^\dagger$,
[Se-Young Yun](https://fbsqkd.github.io)$^\dagger$<br/>
\* equal contribution $&nbsp$ $\dagger$ corresponding author

- **Early-Exiting** dynamically allocates computation paths based on the complexity of generation for each token.
- Conventional framework failed to show actual speedup due to the large number of exit points and state copying mechanism.
- We propose **FREE**, consists of (1) shallow-deep module, (2) synchronized parallel decoding, and (3) adaptive threshold estimator.
- In contrast to conventional approaches, FREE achieved larger inference speedup on extensive generation tasks.

## 🚀 Updates

- [ ] Implement CALM and FREE on decoder-only models
- [x] (24.02.08) Release [**finetuned checkpoints**](https://drive.google.com/drive/folders/1covxgJtIbFgH_xI-sXIuashX2zsY42w_?usp=share_link)
- [x] (24.01.26) Won 🥈Silver award from Samsung Humantech Paper Awards

## Requirements
Install the necessary packages with: 
```
$ pip install -r requirements.txt
```


## Experiments
We experimented with 4 summarization tasks, 1 question answering task, and 1 machine translation task.     
Please see the [scripts](scripts/) and run shell files to train or evaluate on each dataset.    
```bash
$ python run_[TASK_NAME]_[DATASET_NAME].sh
```

### Methods

You can run three early-exiting methods, including Static-Exiting, [CALM](https://proceedings.neurips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf), and our FREE method.    

Here are some important arguments to be considered.     
Please refer [additional_args](util/additional_args.py) for more details.   

#### Training for FREE: 
- `--ouput_hidden_states_decoder True`: return hidden_states from intermediate layers
- `--intermediate_loss_fn shallowdeep_kd_dyna`: use a dynamic distillation loss between shallow and deep models
- `--shallow_exit_layer [int]`: set the number of layers for the shallow model
- `--distill_layer_alpha [float]`: distillation interpolation hyperparameter between CE and KL divergence losses

#### Training for CALM and Static-Exiting: 
- `--ouput_hidden_states_decoder True`: return hidden_states from intermediate layers
- `--intermediate_loss_fn weighted_ce`: use a weighted average loss across all layers
 
#### Evaluation for FREE: 
- `--deploy_scenario True`: this should be always True to use [deploying_[MODEL_NAME].py](models/) for FREE or CALM
- `--use_shallow_deep True`: use shallow-deep module
- `--shallow_exit_layer [int]`: set the number of layers for the shallow model
- `--shallow2deep_conf_type softmax`: set the confidence measure to softmax values
- `--shallow2deep_conf_threshold [float]`: threshold value to decide whether to exit or not in the shallow model
- `--use_adap_threshold True`: use adaptive threshold estimator, where the initial threshold is set to shallow2deep_conf_threshold
 
#### Evaluation for CALM: 
- `--deploy_scenario True`: this should be always True to use [deploying_[MODEL_NAME].py](models/) for FREE or CALM
- `--use_early_exit True`: use conventional early-exiting framework
- `--exit_conf_type softmax`: set the confidence measure to softmax values
- `--exit_conf_threshold [float]`: threshold value to decide whether to exit or not
- `--exit_min_layer [int]`: the minimum number of layers to forward to decide the exiting

#### Evaluation for Static-Exiting: 
- `--static_exit_layer [int]`: set how many layers to use for prediction


### Results

FREE demonstrated robust performance and a larger AUC across various datasets and models, specifically with T5-large and T5-3B.
<p align="center">
<img width="1194" src="https://github.com/raymin0223/fast_robust_early_exit/assets/50742281/d87b9d8c-f774-4111-808d-10df97539b42">
</p>

#### Human-like Summarization Evaluation
We conducted two human-like evaluation methods, Likert scale scoring and pairwise comparison (refer to [this paper](https://arxiv.org/abs/2304.02554)).     
After correctly making input files through [ipynb file](gpt_eval/gpt_eval_example.ipynb), run `bash gpt_eval.sh` with your own OpenAI API_KEY.    
Then, you can get the results by running the last cell in [ipynb file](gpt_eval/gpt_eval_example.ipynb).

### Checkpoints


We share finetuned checkpoints in [google drive](https://drive.google.com/drive/folders/1covxgJtIbFgH_xI-sXIuashX2zsY42w_?usp=share_link).


## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@misc{bae2023fast,
      title={Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding}, 
      author={Sangmin Bae and Jongwoo Ko and Hwanjun Song and Se-Young Yun},
      year={2023},
      eprint={2310.05424},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
- Sangmin Bae: bsmn0223@kaist.ac.kr
- Jongwoo Ko: jongwoo.ko@kaist.ac.kr
