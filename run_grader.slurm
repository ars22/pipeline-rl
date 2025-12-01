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

# Multi-node aware vLLM grader that mirrors the IMOBench multi-node launcher.
# Uses a Ray backend so the job can scale beyond a single node when requested.

module load cuda/12.9
set -x -euo pipefail

source ~/.bashrc
source grader/bin/activate

START_TIME=$(date +%s)
echo "START TIME: $(date)"

usage() {
  echo "Usage: sbatch run_grader.slurm --model MODEL [--dp DATA_PARALLEL] [--tp TENSOR_PARALLEL]" >&2
  exit 1
}

MODEL=""
DP=1
TP=1

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --dp)
      DP="${2:-}"
      shift 2
      ;;
    --tp)
      TP="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

[[ -z "$MODEL" ]] && usage
[[ -z "$DP" ]] && DP=1
[[ -z "$TP" ]] && TP=1

if ! [[ "$DP" =~ ^[0-9]+$ ]] || ! [[ "$TP" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --dp and --tp must be positive integers" >&2
  exit 1
fi

DP=$((DP))
TP=$((TP))
if (( DP < 1 || TP < 1 )); then
  echo "ERROR: --dp and --tp must be >= 1" >&2
  exit 1
fi

NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
  echo "ERROR: SLURM did not provide a node list" >&2
  exit 1
fi

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE=$(echo "$NODES" | head -n1)
if [[ -z "$HEAD_NODE" ]]; then
  echo "ERROR: Unable to determine the head node" >&2
  exit 1
fi

HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -n1 || true)
if [[ -z "$HEAD_NODE_IP" ]]; then
  HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -n1)
fi

if [[ -z "$HEAD_NODE_IP" ]]; then
  echo "ERROR: Unable to determine head node IP address" >&2
  exit 1
fi

count_device_list() {
  local value="${1:-}"
  value="${value// /}"
  value="${value//[[:alpha:]]/}"
  value="${value//[():]/}"
  if [[ -z "$value" ]]; then
    echo 0
    return
  fi
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "$value"
    return
  fi
  local total=0
  local part
  local IFS=","
  read -ra parts <<< "$value"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local start="${BASH_REMATCH[1]}"
      local end="${BASH_REMATCH[2]}"
      if (( end >= start )); then
        total=$((total + end - start + 1))
      fi
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      total=$((total + 1))
    fi
  done
  echo "$total"
}

detect_gpus_per_node() {
  local count=""
  if command -v srun >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
    count=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
      bash -c 'nvidia-smi --query-gpu=index --format=csv,noheader | wc -l' 2>/dev/null | tail -n1 | tr -d '[:space:]') || count=""
  fi

  if ! (( ${count:-0} > 0 )) && command -v nvidia-smi >/dev/null 2>&1; then
    count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]') || count=""
  fi

  if ! (( ${count:-0} > 0 )); then
    if [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
      count=$(count_device_list "${SLURM_GPUS_PER_NODE}")
    elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      count=$(count_device_list "${SLURM_GPUS_ON_NODE}")
    elif [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
      count=$(count_device_list "${SLURM_STEP_GPUS}")
    elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
      count=$(count_device_list "${SLURM_JOB_GPUS}")
    fi
  fi

  if ! (( ${count:-0} > 0 )); then
    count=$(count_device_list "${CUDA_VISIBLE_DEVICES:-}")
  fi

  echo "${count:-0}"
}

NUM_GPUS_PER_NODE=$(detect_gpus_per_node)

if ! (( NUM_GPUS_PER_NODE > 0 )); then
  echo "ERROR: Unable to detect GPUs per node" >&2
  exit 1
fi

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
REQUIRED_GPUS=$((DP * TP))

if (( REQUIRED_GPUS > TOTAL_GPUS )); then
  echo "ERROR: Requested DP*TP=${REQUIRED_GPUS} exceeds available GPUs (${TOTAL_GPUS})" >&2
  exit 1
fi

LOG_DIR=/fsx/h4/logs
mkdir -p "$LOG_DIR"
VLLM_LOG="${LOG_DIR}/imo-grader-${SLURM_JOB_ID}-vllm.log"
RAY_PORT=${RAY_PORT:-6379}
VLLM_PORT=${VLLM_PORT:-8000}
CPUS_PER_NODE=${SLURM_CPUS_PER_TASK:-$(nproc)}

export MODEL
export DP
export TP
export VLLM_API_KEY="grader"

echo "================================================================"
echo "LLM Grader multi-node launch configuration"
echo "Model: $MODEL"
echo "DP: $DP | TP: $TP | Requested GPUs: $REQUIRED_GPUS"
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Head node: $HEAD_NODE (IP: $HEAD_NODE_IP)"
echo "All nodes:"
echo "$NODES"
echo "vLLM log: $VLLM_LOG"
echo "================================================================"

CLEANED_UP=0
SRUN_HEAD_PID=""
SRUN_WORKER_PID=""
VLLM_PID=""
RAY_CLUSTER_STARTED=0

cleanup() {
  local exit_code=${1:-0}
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "Stopping vLLM server (PID: $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi

  if [[ "$RAY_CLUSTER_STARTED" -eq 1 ]]; then
    echo "Stopping Ray cluster..."
    srun --nodes=$NUM_NODES --ntasks=$NUM_NODES ray stop --force 2>/dev/null || true
  fi

  if [[ -n "$SRUN_HEAD_PID" ]] && kill -0 "$SRUN_HEAD_PID" 2>/dev/null; then
    kill "$SRUN_HEAD_PID" 2>/dev/null || true
  fi

  if [[ -n "$SRUN_WORKER_PID" ]] && kill -0 "$SRUN_WORKER_PID" 2>/dev/null; then
    kill "$SRUN_WORKER_PID" 2>/dev/null || true
  fi

  echo "Cleanup complete (exit code: $exit_code)"
}

trap 'cleanup $?' EXIT
trap 'cleanup 130; exit 130' SIGINT
trap 'cleanup 143; exit 143' SIGTERM

export RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

echo "Cleaning up any stale Ray sessions..."
srun --nodes=$NUM_NODES --ntasks=$NUM_NODES ray stop --force 2>/dev/null || true
sleep 3

echo "Starting Ray head node on $HEAD_NODE ($HEAD_NODE_IP)..."
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
  ray start --head \
  --node-ip-address="$HEAD_NODE_IP" \
  --port=$RAY_PORT \
  --num-cpus=$CPUS_PER_NODE \
  --num-gpus=$NUM_GPUS_PER_NODE \
  --block &
SRUN_HEAD_PID=$!
RAY_CLUSTER_STARTED=1

echo "Waiting for Ray head node initialization..."
sleep 20

if (( NUM_NODES > 1 )); then
  WORKER_COUNT=$((NUM_NODES - 1))
  echo "Starting $WORKER_COUNT Ray worker node(s)..."
  srun --nodes=$WORKER_COUNT --ntasks=$WORKER_COUNT --exclude="$HEAD_NODE" \
    ray start --address="$RAY_ADDRESS" \
    --num-cpus=$CPUS_PER_NODE \
    --num-gpus=$NUM_GPUS_PER_NODE \
    --block &
  SRUN_WORKER_PID=$!
  echo "Waiting for Ray workers to connect..."
  sleep 30
fi

echo "Waiting for Ray cluster to register $TOTAL_GPUS GPUs..."
for attempt in {1..60}; do
  CURRENT_GPUS=$(python3 - <<'PY'
import logging
import os
import ray
try:
    ray.init(address=os.environ["RAY_ADDRESS"], logging_level=logging.ERROR)
    print(int(ray.cluster_resources().get("GPU", 0)))
except Exception:
    print(0)
PY
)
  CURRENT_GPUS=$(echo "$CURRENT_GPUS" | tail -n1 | tr -d '[:space:]')

  echo "Ray cluster status: ${CURRENT_GPUS:-0} / $TOTAL_GPUS GPUs ready (attempt $attempt/60)"

  if [[ "${CURRENT_GPUS:-0}" -ge "$TOTAL_GPUS" ]]; then
    echo "✓ Ray cluster ready with $CURRENT_GPUS GPUs."
    break
  fi

  if [[ "$attempt" -eq 60 ]]; then
    echo "ERROR: Ray cluster failed to report $TOTAL_GPUS GPUs within timeout." >&2
    exit 1
  fi

  sleep 5
done

echo "Launching vLLM server on Ray backend..."
echo "Server command log: $VLLM_LOG"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --distributed-executor-backend ray \
  --data-parallel-size $DP \
  --tensor-parallel-size $TP \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests \
  &> "$VLLM_LOG" &
VLLM_PID=$!

echo "vLLM server started (PID: $VLLM_PID). Listening on ${HEAD_NODE_IP}:${VLLM_PORT}."
echo "Press Ctrl+C or cancel the Slurm job to stop the grader."

wait "$VLLM_PID"
