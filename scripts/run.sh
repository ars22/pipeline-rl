#!/bin/bash

# sleep 259200 
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29517
python -m pipelinerl.launch --config-name=pope output_dir=/project/flame/asetlur/pipeline-rl/results/prl-pope-16k-8a8f-grpo-n4-d

# accelerate launch --num_processes 2 -m pipelinerl.launch --config-name=pope output_dir=/project/flame/asetlur/pipeline-rl/results/prl-pope-16k-8a8f-grpo-n4