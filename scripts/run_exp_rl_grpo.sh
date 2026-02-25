#!/bin/bash

python -m pipelinerl.launch \
--config-name=exp_rl_grpo \
wandb.wandb_project_name=prl_exp_rl_qwen4b_thinking \
output_dir=./results/prl-exp-rl-grpo-32k-16a-grpo
