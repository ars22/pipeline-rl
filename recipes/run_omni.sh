#!/bin/bash

python -m pipelinerl.launch \
--config-name=omni_genrm \
output_dir=results/smooth_qwen3-4b \
wandb.wandb_project_name=omni-cohen-genrm \