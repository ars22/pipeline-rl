#!/bin/bash
#SBATCH --job-name=imo-grader
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/12.9
set -x -e

source ~/.bashrc
source grader/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --dp)
      DP="${2:-1}"
      shift 2
      ;;
    --tp)
      TP="${2:-1}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: sbatch run_grader.slurm [--model MODEL] [--dp DP] [--tp TP]"
      exit 1
      ;;
  esac
done

export MODEL
export DP
export TP

terminate_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "Stopping vLLM server (PID $SERVER_PID)"
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap 'terminate_server; exit 130' SIGINT
trap 'terminate_server; exit 143' SIGTERM

echo "Starting vllm server with model: $MODEL, DP: $DP, TP: $TP"
vllm serve $MODEL --data-parallel-size $DP --tensor-parallel-size $TP \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu_memory_utilization 0.9 &
SERVER_PID=$!

wait "$SERVER_PID"
