# PathFinder: Towards Self-Supervised Evolution LLM without Human Feedbacks

Pathfinder is a Fully Self-Supervised Language model tuning technique.

In the quest to leverage the vast amounts of unlabeled data available, we introduce PathFinder, a novel, fully self-supervised fine-tuning algorithm for Large Language Models (LLMs). PathFinder necessitates only the LLM's inherent ability to follow basic instructions, enabling it to exploit any unlabeled dataset of questions, which can also be autonomously sampled by the model itself. This method advances the model's comprehension and generative capabilities without relying on labeled data, thus paving the way for fully autonomous and efficient LLM evolution.

## Usage
Install conda dependencies in `conda.yaml`
configure Accelerate like `accelerate.yaml`

```
cd LLaMa-Factory
accelerate launch src/train_bash.py     --stage sft     --do_train     --model_name_or_path google/gemma-2b-it     --dataset mathinstruct    --template gemma     --finetuning_type lora --lora_target all     --output_dir path_to_math_checkpoint      --per_device_train_batch_size 1     --gradient_accumulation_steps 2     --lr_scheduler_type cosine     --logging_steps 1     --save_steps 50     --learning_rate 5e-5     --num_train_epochs 1     --plot_loss     --bf16  --upcast_layernorm=true --ddp_find_unused_parameters=false --flash_attn --neftune_noise_alpha 5 --preprocessing_num_workers 8 --overwrite_output_dir
```

## Credit

Zhang Di @ FDU