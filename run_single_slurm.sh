#!/bin/bash
#SBATCH --job-name=pipe-rl
#SBATCH --partition=flame # Or your desired partition
#SBATCH --nodes=2           # Request exactly 2 nodes
#SBATCH --ntasks-per-node=1 # Run one main task per node
#SBATCH --gres=gpu:8        # 8 GPUs per node
#SBATCH --cpus-per-task=16  # 16 CPUs per node (ensure nodes have this many cores available)
#SBATCH --mem=512G         # 1024G RAM per node (ensure nodes have this much memory)
#SBATCH --time=47:59:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err  # Good practice for separate error logs
#SBATCH --qos=flame-64gpu_qos
#SBATCH --account=aviralku
#SBATCH --exclude=orchard-flame-17

JOB_WORKING_DIR="/home/asetlur/PipelineRL"
JOB_SCRIPT_NAME="$JOB_WORKING_DIR/scripts/run.sh"

# --- Setup ---
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs per node: $SLURM_GPUS_ON_NODE" # Verify Slurm is parsing --gres correctly
echo "CPUs per task/node: $SLURM_CPUS_PER_TASK"

cd $JOB_WORKING_DIR
sh -c "exec bash $JOB_SCRIPT_NAME"