#!/bin/bash
#SBATCH --job-name=imo-grader
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

HEALTH_CHECK_URL=${HEALTH_CHECK_URL:-http://127.0.0.1:8000/health}
HEALTH_CHECK_RETRIES=${HEALTH_CHECK_RETRIES:-60}
HEALTH_CHECK_DELAY=${HEALTH_CHECK_DELAY:-10}

health_check() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the health check but was not found in PATH" >&2
    return 1
  fi

  echo "Waiting for vLLM server health at $HEALTH_CHECK_URL"
  for (( i=1; i<=HEALTH_CHECK_RETRIES; i++ )); do
    if curl -fsS "$HEALTH_CHECK_URL" >/dev/null; then
      echo "vLLM health check succeeded on attempt $i"
      return 0
    fi
    echo "Health check attempt $i/$HEALTH_CHECK_RETRIES failed; retrying in ${HEALTH_CHECK_DELAY}s"
    sleep "$HEALTH_CHECK_DELAY"
  done

  echo "vLLM health check failed after $HEALTH_CHECK_RETRIES attempts" >&2
  return 1
}

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
vllm serve $MODEL --data-parallel-size $DP --tensor-parallel-size $TP &
SERVER_PID=$!

if ! health_check; then
  echo "Health check failed; terminating vLLM server" >&2
  terminate_server
  exit 1
fi

wait "$SERVER_PID"
