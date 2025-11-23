#!/bin/bash
SOURCE_EXP="results/smooth_qwen3-4b"
OUTPUT_DIR="results/debug_finetune_demo"

if [ ! -d "$SOURCE_EXP" ]; then
    echo "Error: Source experiment $SOURCE_EXP does not exist."
    exit 1
fi

echo "Starting debug finetune run..."
echo "Reading streams from: $SOURCE_EXP"
echo "Writing output to: $OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0
python -m pipelinerl.launch \
    --config-name=omni_genrm \
    output_dir=$OUTPUT_DIR \
    debug.mode=finetune \
    debug.streams_from=$SOURCE_EXP \
    finetune.train_batch_size=1 \
    finetune.seq_parallel=1 \
    finetune.gradient_accumulation_passes=4 \
    finetune.rl.kl_coef=0 \
    world.actor_fraction=0 \
    world.preprocessor_fraction=0 \
    world.genrm_fraction=0 \
    world.finetune_fraction=1
