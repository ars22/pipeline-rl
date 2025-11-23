#!/bin/bash

task_id=$1

model_paths=(
    Qwen/Qwen3-4B-Instruct-2507
)

model_names=(
    Instruct
)

tasks=(
    olympiads-proof
)

hf_train_datasets=(
    hf-imo-colab/olympiads-proof-schema
)

hf_train_splits=(
    train
)

hf_test_datasets=(
    hf-imo-colab/olympiads-proof-schema-benchmark
)

hf_test_splits=(
    IMO2025
)


model_path=${model_paths[$task_id]}
model_name=${model_names[$task_id]}
task=${model_name}-${tasks[$task_id]}
train_dataset_path=${hf_train_datasets[$task_id]}
train_split=${hf_train_splits[$task_id]}
test_dataset_path=${hf_test_datasets[$task_id]}
test_split=${hf_test_splits[$task_id]}

export OPENAI_BASE_URL=""
export OPENAI_API_KEY=""


export WANDB_API_KEY=""
export WANDB_ENTITY=

python -m pipelinerl.launch --config-name=hf-imo-colab-proof output_dir=models/${task}_m2 "train_dataset_names=[{hub_id: ${train_dataset_path}, split: ${train_split}}]" "test_dataset_names=[{hub_id: ${test_dataset_path}, split: ${test_split}}]" finetune.hub_model_id=hf-imo-colab/${task}