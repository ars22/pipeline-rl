#!/bin/bash

python -m pipelinerl.launch \
--config-name=pope_hard_w_gemin_sol \
wandb.wandb_project_name=prl_pope_hard_w_gemini_sol_qwen4b_instruct \
output_dir=/hai/scratch/ziyxiang/pipeline-rl/results/prl-pope-hard-w-gemini-sol-16k-8a8f-grpo-n4-d