---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft

This model is a fine-tuned version of [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on the news_finetune_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3492

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.4779        | 0.1481 | 100  | 0.4030          |
| 0.3938        | 0.2963 | 200  | 0.3865          |
| 0.5203        | 0.4444 | 300  | 0.3678          |
| 0.4821        | 0.5926 | 400  | 0.3523          |
| 0.3828        | 0.7407 | 500  | 0.3403          |
| 0.4154        | 0.8889 | 600  | 0.3397          |
| 0.2703        | 1.0370 | 700  | 0.3361          |
| 0.2253        | 1.1852 | 800  | 0.3383          |
| 0.2677        | 1.3333 | 900  | 0.3323          |
| 0.2723        | 1.4815 | 1000 | 0.3259          |
| 0.3145        | 1.6296 | 1100 | 0.3277          |
| 0.2413        | 1.7778 | 1200 | 0.3238          |
| 0.292         | 1.9259 | 1300 | 0.3214          |
| 0.1336        | 2.0741 | 1400 | 0.3490          |
| 0.1882        | 2.2222 | 1500 | 0.3506          |
| 0.1967        | 2.3704 | 1600 | 0.3506          |
| 0.2248        | 2.5185 | 1700 | 0.3511          |
| 0.1419        | 2.6667 | 1800 | 0.3490          |
| 0.1606        | 2.8148 | 1900 | 0.3498          |
| 0.1919        | 2.9630 | 2000 | 0.3491          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.48.2
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0