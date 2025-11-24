#!/bin/bash

python -m pipelinerl.launch \
--config-name=omni_genrm \
finetune.rl.entropy_bonus=0.001 \
finetune.rl.epsilon=0.1 \
output_dir=results/smooth_qwen3-4b_e0.1_ent1e-4 \
wandb.wandb_project_name=omni-cohen-genrm