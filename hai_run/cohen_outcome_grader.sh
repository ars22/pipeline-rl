#!/bin/bash
#SBATCH --job-name=cohen-outcome-grader
#SBATCH --account=ingrai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# project dir
JOB_WORKING_DIR="/hai/scratch/ziyxiang/pipeline-rl"

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

# Get node information
export WORLD_SIZE=$SLURM_NNODES
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# Build hostlist for srun
HOSTLIST=$(printf '%s\n' "${nodes_array[@]}" | paste -sd, -)

# Collect "hostname ip" pairs from the nodes
declare -A IP_OF
while read -r host ip; do
  [[ -n "$host" ]] && IP_OF["$host"]="$ip"
done < <(
  srun -w "$HOSTLIST" --ntasks-per-node=1 bash -c '
    h=$(hostname -s)
    ip=$(hostname -I | tr " " "\n" | grep -m1 -E "^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$")
    echo "$h $ip"
  '
)

# Fill the indexed array in the same order as nodes_array
ip_addr=()
for h in "${nodes_array[@]}"; do
  ip_addr+=("${IP_OF[$h]}")
done

# Export addresses
export ALL_ADDR="$(IFS=,; echo "${ip_addr[*]}")"
if ((${#nodes_array[@]} > 0)); then
  export MASTER_ADDR="${ip_addr[0]}"
  export MASTER_PORT=6379
else
  echo "nodes_array is empty; cannot set MASTER_ADDR" >&2
  exit 1
fi

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "Nodes allocated: ${nodes_array[@]}"
echo "Host to IP mapping:"
for i in "${!nodes_array[@]}"; do
  printf "  %s -> %s\n" "${nodes_array[$i]}" "${ip_addr[$i]}"
done
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ALL_ADDR=$ALL_ADDR"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "============================================"

cd $JOB_WORKING_DIR

# Run on all nodes
srun -w "$HOSTLIST" --ntasks-per-node=1 \
  bash -lc '    
    source /hai/scratch/ziyxiang/miniconda3/etc/profile.d/conda.sh
    conda activate prl

    # Fix AMD/NVIDIA env conflict
    unset ROCR_VISIBLE_DEVICES

    # Ray settings
    export RAY_TMPDIR=~/ray_tmp
    export RAY_GCS_PORT=6388
    export RAY_DASHBOARD_PORT=8278
    export RAY_DISABLE_DASHBOARD_AGENT=1

    # Multi-node settings
    export RANK=$SLURM_NODEID
    export WORLD_SIZE=$SLURM_NNODES
    export ALL_ADDR='"$ALL_ADDR"'
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'

    echo "[$(hostname -s)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR"

    cd '"$JOB_WORKING_DIR"'

    python -m pipelinerl.launch_pr --config-name=cohen_llm_grader \
      output_dir=./results/cohen-outcome-grader \
      world.actor_fraction=8 \
      world.preprocessor_fraction=0 \
      world.finetune_fraction=8
  '
