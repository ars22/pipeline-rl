#!/bin/bash

python -m pipelinerl.launch \
--config-name=int_train \
wandb.wandb_project_name=prl_int_train_qwen4b_instruct \
output_dir=./results/prl-int-train-16k-8a8f-grpo-n4-d