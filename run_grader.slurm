#!/bin/bash
#SBATCH --job-name=imo-grader
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --error=/fsx/h4/logs/%x-%j.err
#SBATCH --time=3-0:00:00

# Multi-node grader launch script that uses Ray for distributed vLLM serving

# compilation does not play nice with Ray
export TORCH_COMPILE_DISABLE=1

module load cuda/12.9
set -x -e

source ~/.bashrc
source grader/bin/activate

START_TIME=$(date +%s)
echo "START TIME: $(date)"

usage() {
  echo "Usage: sbatch run_grader.slurm --model MODEL [--data-parallel-size DATA_PARALLEL] [--tensor-parallel-size TENSOR_PARALLEL] [--ray-port PORT] [--vllm-port PORT] [--max-num-batched-tokens TOKENS] [--max-num-seqs SEQS] [--max-model-len TOKENS] [--gpu-memory-utilization FRACTION]" >&2
  exit 1
}

MODEL=""
DATA_PARALLEL_SIZE=1
TENSOR_PARALLEL_SIZE=1
RAY_PORT=${RAY_PORT:-6379}
VLLM_PORT=${VLLM_PORT:-8000}
MAX_NUM_BATCHED_TOKENS=8192
MAX_NUM_SEQS=16
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.85

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --data-parallel-size)
      DATA_PARALLEL_SIZE="${2:-}"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="${2:-}"
      shift 2
      ;;
    --ray-port)
      RAY_PORT="${2:-}"
      shift 2
      ;;
    --vllm-port)
      VLLM_PORT="${2:-}"
      shift 2
      ;;
    --max-num-batched-tokens)
      MAX_NUM_BATCHED_TOKENS="${2:-}"
      shift 2
      ;;
    --max-num-seqs)
      MAX_NUM_SEQS="${2:-}"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="${2:-}"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

[[ -z "$MODEL" ]] && usage
[[ -z "$DATA_PARALLEL_SIZE" ]] && DATA_PARALLEL_SIZE=1
[[ -z "$TENSOR_PARALLEL_SIZE" ]] && TENSOR_PARALLEL_SIZE=1
[[ -z "$RAY_PORT" ]] && RAY_PORT=6379
[[ -z "$VLLM_PORT" ]] && VLLM_PORT=8000

if ! [[ "$DATA_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --data-parallel-size and --tensor-parallel-size must be positive integers" >&2
  exit 1
fi

if ! [[ "$RAY_PORT" =~ ^[0-9]+$ ]] || ! [[ "$VLLM_PORT" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --ray-port and --vllm-port must be integers" >&2
  exit 1
fi

if ! [[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --max-num-batched-tokens must be a positive integer" >&2
  exit 1
fi

if ! [[ "$MAX_NUM_SEQS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --max-num-seqs must be a positive integer" >&2
  exit 1
fi

if ! [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --max-model-len must be a positive integer" >&2
  exit 1
fi

if ! [[ "$GPU_MEMORY_UTILIZATION" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: --gpu-memory-utilization must be a positive number" >&2
  exit 1
fi

DATA_PARALLEL_SIZE=$((DATA_PARALLEL_SIZE))
TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE))
RAY_PORT=$((RAY_PORT))
VLLM_PORT=$((VLLM_PORT))
MAX_NUM_BATCHED_TOKENS=$((MAX_NUM_BATCHED_TOKENS))
MAX_NUM_SEQS=$((MAX_NUM_SEQS))
MAX_MODEL_LEN=$((MAX_MODEL_LEN))

if (( DATA_PARALLEL_SIZE < 1 || TENSOR_PARALLEL_SIZE < 1 )); then
  echo "ERROR: --data-parallel-size and --tensor-parallel-size must be >= 1" >&2
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

HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b" | head -1 || true)
if [[ -z "$HEAD_NODE_IP" ]]; then
  HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -n1)
fi

if [[ -z "$HEAD_NODE_IP" ]]; then
  echo "ERROR: Unable to determine head node IP address" >&2
  exit 1
fi

NUM_GPUS_PER_NODE=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" nvidia-smi -L | wc -l)

if ! (( NUM_GPUS_PER_NODE > 0 )); then
  echo "ERROR: Unable to detect GPUs per node" >&2
  exit 1
fi

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
REQUIRED_GPUS=$((DATA_PARALLEL_SIZE * TENSOR_PARALLEL_SIZE))

if (( REQUIRED_GPUS > TOTAL_GPUS )); then
  echo "ERROR: Requested data-parallel-size * tensor-parallel-size = ${REQUIRED_GPUS} exceeds available GPUs (${TOTAL_GPUS})" >&2
  exit 1
fi

if (( DATA_PARALLEL_SIZE < NUM_NODES )); then
  echo "ERROR: Requested data-parallel-size (${DATA_PARALLEL_SIZE}) must be >= number of nodes (${NUM_NODES})" >&2
  exit 1
fi

if (( DATA_PARALLEL_SIZE % NUM_NODES != 0 )); then
  echo "ERROR: Requested data-parallel-size (${DATA_PARALLEL_SIZE}) must be divisible by number of nodes (${NUM_NODES}) to evenly distribute replicas" >&2
  exit 1
fi

MAX_DATA_PARALLEL_PER_NODE=$((NUM_GPUS_PER_NODE / TENSOR_PARALLEL_SIZE))
if (( MAX_DATA_PARALLEL_PER_NODE == 0 )); then
  echo "ERROR: Tensor parallel size (${TENSOR_PARALLEL_SIZE}) exceeds GPUs per node (${NUM_GPUS_PER_NODE})" >&2
  exit 1
fi

DATA_PARALLEL_PER_NODE=$((DATA_PARALLEL_SIZE / NUM_NODES))
if (( DATA_PARALLEL_PER_NODE > MAX_DATA_PARALLEL_PER_NODE )); then
  echo "ERROR: Data parallel replicas per node (${DATA_PARALLEL_PER_NODE}) exceeds capacity (${MAX_DATA_PARALLEL_PER_NODE}) for tensor-parallel-size=${TENSOR_PARALLEL_SIZE}" >&2
  exit 1
fi

LOG_DIR=/fsx/h4/logs
mkdir -p "$LOG_DIR"
VLLM_LOG="${LOG_DIR}/imo-grader-${SLURM_JOB_ID}-vllm.log"

CPUS_PER_NODE=88

export MODEL
export DATA_PARALLEL_SIZE
export TENSOR_PARALLEL_SIZE
export VLLM_API_KEY="grader"
export VLLM_HOST_IP="$HEAD_NODE_IP"

echo "================================================================"
echo "LLM Grader multi-node launch configuration"
echo "Model: $MODEL"
echo "Data parallel size: $DATA_PARALLEL_SIZE (per node: $DATA_PARALLEL_PER_NODE) | Tensor parallel size: $TENSOR_PARALLEL_SIZE | Requested GPUs: $REQUIRED_GPUS"
echo "Nodes: $NUM_NODES"
echo "GPUs per node: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "Head node: $HEAD_NODE (IP: $HEAD_NODE_IP)"
echo "Ray port: $RAY_PORT"
echo "vLLM port: $VLLM_PORT"
echo "CPUs per node: $CPUS_PER_NODE"
echo "Max num batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "Max num seqs: $MAX_NUM_SEQS"
echo "Max model len: $MAX_MODEL_LEN"
echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
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
  CURRENT_GPUS=$(python3 -c "import ray; ray.init(); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo "0")

  echo "Ray cluster status: $CURRENT_GPUS / $TOTAL_GPUS GPUs ready (attempt $attempt/60)"

  if [ "$CURRENT_GPUS" -ge "$TOTAL_GPUS" ]; then
    echo "✓ Ray cluster ready with $TOTAL_GPUS GPUs!"
    break
  fi

  if [ $attempt -eq 60 ]; then
    echo "ERROR: Ray cluster failed to report $TOTAL_GPUS GPUs within timeout." >&2
    exit 1
  fi

  sleep 5
done

echo "Launching vLLM server on Ray backend..."
echo "Server command log: $VLLM_LOG"

# Optimised for gpt-oss models: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
# Notes:
# - async scheduling not compatible with Ray backend
# - we use a large max-num-batched-tokens to improve throughput for long inputs / outputs
# - setting gpu-memory-utilization to values >0.85 can lead to OOMs
# TODO: expose more of these as config args.
vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --distributed-executor-backend ray \
  --data-parallel-backend ray \
  --data-parallel-address "$HEAD_NODE_IP" \
  --data-parallel-size $DATA_PARALLEL_SIZE \
  --data-parallel-size-local $DATA_PARALLEL_PER_NODE \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --max-cudagraph-capture-size 2048 \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-num-seqs $MAX_NUM_SEQS \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
  &> "$VLLM_LOG" &
VLLM_PID=$!

echo "vLLM server started (PID: $VLLM_PID). Listening on ${HEAD_NODE_IP}:${VLLM_PORT}."
echo "Press Ctrl+C or cancel the Slurm job to stop the grader."

wait "$VLLM_PID"
