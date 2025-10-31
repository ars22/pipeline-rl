#!/bin/bash
#SBATCH --job-name=pipe-rl
#SBATCH --partition=flame # Or your desired partition
#SBATCH --nodes=2           # Request exactly 2 nodes
#SBATCH --ntasks-per-node=1 # Run one main task per node
#SBATCH --gres=gpu:8        # 8 GPUs per node
#SBATCH --cpus-per-task=8  # 16 CPUs per node (ensure nodes have this many cores available)
#SBATCH --mem=300G         # 1024G RAM per node (ensure nodes have this much memory)
#SBATCH --time=47:59:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Good practice for separate error logs
#SBATCH --account=aviralku
#SBATCH --qos=flame-64gpu_qos


export WORLD_SIZE=$SLURM_NTASKS
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

# Export stuff
export ALL_ADDR="$(IFS=,; echo "${ip_addr[*]}")"  # comma-separated (optional)
if ((${#nodes_array[@]} > 0)); then
  export MASTER_ADDR="${ip_addr[0]}"
  export MASTER_PORT=6379
else
  echo "nodes_array is empty; cannot set MASTER_ADDR" >&2
  exit 1
fi


# Define the absolute path to the working directory for the job
JOB_WORKING_DIR="/home/asetlur/PipelineRL"
JOB_SCRIPT_NAME="$JOB_WORKING_DIR/scripts/run.sh"


# Print some info
echo "WORLD_SIZE=$WORLD_SIZE"
echo "Nodes allocated: ${nodes_array[@]}"
echo "Host to IP mapping:"
for i in "${!nodes_array[@]}"; do
  printf "%s %s\n" "${nodes_array[$i]}" "${ip_addr[$i]}"
done
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ALL_ADDR=$ALL_ADDR"
echo "--------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE" 
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"

cd $JOB_WORKING_DIR

srun -w "$HOSTLIST" --ntasks-per-node=1 \
  bash -lc '
    export RANK=$SLURM_NODEID
    export WORLD_SIZE=$SLURM_NNODES
    export ALL_ADDR=$ALL_ADDR
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    echo "[$(hostname -s)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT ALL_ADDR=$ALL_ADDR"
    python -m pipelinerl.launch --config-name=pope \
      output_dir=/project/flame/asetlur/pipeline-rl/results/prl-pope-16k-8a8f-grpo-n4-r2
    '


# sh -c "exec bash $JOB_SCRIPT_NAME"






# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)

# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# head_node_ip=$(echo $head_node_ip | awk '{print $1}') # Clean up potential extra output

# Validate IP address was obtained
# if [ -z "$head_node_ip" ]; then
#     echo "ERROR: Failed to obtain head node IP address."
#     exit 1
# fi

# export MASTER_ADDR=$head_node_ip