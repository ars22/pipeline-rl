import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

_GRADER_JOB_ID: str | None = None
_GRADER_CLEANUP_REGISTERED = False
_GRADER_RESERVED_VLLM_KEYS = {
    "num-nodes",
    "data-parallel-size",
    "tensor-parallel-size",
    "ray-port",
    "vllm-port",
    "max-num-batched-tokens",
    "max-num-seqs",
    "max-model-len",
    "gpu-memory-utilization",
}
_GRADER_RESERVED_VLLM_DEFAULTS = {
    "num-nodes": 1,
    "data-parallel-size": 1,
    "tensor-parallel-size": 1,
    "ray-port": 6379,
    "vllm-port": 8000,
    "max-num-batched-tokens": 8192,
    "max-num-seqs": 16,
    "max-model-len": 32768,
    "gpu-memory-utilization": 0.85,
}
_GRADER_BLOCKED_EXTRA_VLLM_KEYS = {
    "model",
    "host",
    "port",
    "distributed-executor-backend",
    "data-parallel-backend",
    "data-parallel-address",
    "data-parallel-size-local",
}


def _cancel_llm_grader_job():
    global _GRADER_JOB_ID
    if not _GRADER_JOB_ID:
        return
    job_id = _GRADER_JOB_ID
    try:
        subprocess.run(["scancel", job_id], capture_output=True, text=True, check=True)
        logger.info(f"Cancelled local LLM grader Slurm job {job_id}")
    except subprocess.CalledProcessError as exc:
        logger.warning(f"Failed to cancel LLM grader job {job_id}: {exc}")
    finally:
        _GRADER_JOB_ID = None


def _handle_exit_signal(signum, _frame):
    logger.info(f"Received signal {signum}, cancelling LLM grader job before exit")
    _cancel_llm_grader_job()
    sys.exit(128 + signum)


def _ensure_grader_cleanup_hooks():
    global _GRADER_CLEANUP_REGISTERED
    if _GRADER_CLEANUP_REGISTERED:
        return
    atexit.register(_cancel_llm_grader_job)
    signal.signal(signal.SIGTERM, _handle_exit_signal)
    signal.signal(signal.SIGINT, _handle_exit_signal)
    _GRADER_CLEANUP_REGISTERED = True


def _wait_for_slurm_nodes(job_id: str, timeout: int = 300, poll_interval: int = 5) -> str:
    """Poll Slurm until a job is assigned to a node."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%N"],
            capture_output=True,
            text=True,
            check=True,
        )
        nodes = result.stdout.strip()
        if nodes and nodes not in {"(null)", "None"}:
            return nodes
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for node assignment for Slurm job {job_id}")


def _expand_slurm_node_list(nodes: str) -> list[str]:
    """Expand a Slurm node-list string into concrete hostnames."""
    try:
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodes],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Unable to expand Slurm node list {nodes}: {exc}") from exc

    hostnames = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not hostnames:
        raise RuntimeError(f"Slurm returned no hostnames for node list {nodes}")
    return hostnames


def _wait_for_vllm_health(url: str, retries: int = 60, delay: int = 10, timeout: int = 5) -> None:
    """Poll the vLLM health endpoint until it responds successfully."""
    logger.info("Waiting for vLLM server health at %s", url)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
                if 200 <= response.status < 300:
                    logger.info("vLLM health check succeeded on attempt %s", attempt)
                    return
        except Exception as exc:  # noqa: BLE001 - broad catch to keep retrying
            last_error = exc
        logger.info(
            "Health check attempt %s/%s failed; retrying in %ss",
            attempt,
            retries,
            delay,
        )
        time.sleep(delay)
    raise RuntimeError(f"vLLM health check failed after {retries} attempts: {last_error}")


def _to_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    if isinstance(config, dict):
        return dict(config)
    return dict(config)


def _normalize_grader_vllm_kwargs(vllm_kwargs: Any | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    original_keys: dict[str, str] = {}
    for raw_key, value in _to_dict(vllm_kwargs).items():
        key = str(raw_key)
        normalized_key = key.replace("_", "-")
        if normalized_key in normalized and key != original_keys[normalized_key]:
            raise ValueError(
                "Conflicting grader vLLM arg keys "
                f"{original_keys[normalized_key]!r} and {key!r}; both normalize to {normalized_key!r}"
            )
        normalized[normalized_key] = value
        original_keys[normalized_key] = key
    return normalized


def _serialize_grader_vllm_arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _build_grader_extra_vllm_args(vllm_kwargs: dict[str, Any]) -> list[str]:
    blocked = sorted(key for key in vllm_kwargs if key in _GRADER_BLOCKED_EXTRA_VLLM_KEYS)
    if blocked:
        blocked_args = ", ".join(f"--{key}" for key in blocked)
        raise ValueError(f"Unsupported llm_grader.vllm_kwargs passthrough arg(s): {blocked_args}")

    cli_args: list[str] = []
    for key, value in vllm_kwargs.items():
        cli_args.append(f"--{key}")
        if value not in [None, ""]:
            cli_args.append(_serialize_grader_vllm_arg_value(value))
    return cli_args


def _has_grader_vllm_arg(cli_args: list[str], arg_name: str) -> bool:
    option = f"--{arg_name}"
    return any(arg == option or arg.startswith(f"{option}=") for arg in cli_args)


def _parse_grader_vllm_kwargs(vllm_kwargs: Any | None) -> tuple[dict[str, Any], list[str]]:
    normalized_kwargs = _normalize_grader_vllm_kwargs(vllm_kwargs)
    reserved = dict(_GRADER_RESERVED_VLLM_DEFAULTS)
    for key in _GRADER_RESERVED_VLLM_KEYS:
        if key in normalized_kwargs:
            reserved[key] = normalized_kwargs.pop(key)
    return reserved, _build_grader_extra_vllm_args(normalized_kwargs)


def start_llm_grader(name: str, vllm_kwargs: Any | None = None, namespace: str = "HuggingFaceH4", timeout=900):
    reserved_kwargs, extra_vllm_args = _parse_grader_vllm_kwargs(vllm_kwargs)
    if not _has_grader_vllm_arg(extra_vllm_args, "api-server-count"):
        # vLLM 0.17.x defaults api_server_count to data_parallel_size for internal
        # load balancing, which has been unreliable for the grader's multi-node
        # launches. Keep a single frontend by default unless the config opts in.
        extra_vllm_args = ["--api-server-count", "1", *extra_vllm_args]
    num_nodes = int(reserved_kwargs["num-nodes"])
    data_parallel_size = int(reserved_kwargs["data-parallel-size"])
    tensor_parallel_size = int(reserved_kwargs["tensor-parallel-size"])
    ray_port = int(reserved_kwargs["ray-port"])
    vllm_port = int(reserved_kwargs["vllm-port"])
    max_num_batched_tokens = reserved_kwargs["max-num-batched-tokens"]
    max_num_seqs = reserved_kwargs["max-num-seqs"]
    max_model_len = reserved_kwargs["max-model-len"]
    gpu_memory_util = reserved_kwargs["gpu-memory-utilization"]
    if "/" in name:
        logger.info(f"Starting local LLM grader {name}...")
        job_name = None
        current_job_id = os.environ.get("SLURM_JOB_ID")
        if current_job_id:
            job_name = f"{current_job_id}-grader"
        cmd = [
            "sbatch",
            "--parsable",
            f"--nodes={num_nodes}",
        ]
        if job_name:
            cmd.append(f"--job-name={job_name}")
        cmd += [
            "run_grader.slurm",
            "--model",
            name,
            "--data-parallel-size",
            str(data_parallel_size),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--ray-port",
            str(ray_port),
            "--vllm-port",
            str(vllm_port),
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--max-num-seqs",
            str(max_num_seqs),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_util),
        ]
        if extra_vllm_args:
            cmd.extend(["--", *extra_vllm_args])
        submission = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = submission.stdout.strip().split(";")[0]
        if not job_id:
            raise RuntimeError("sbatch did not return a job id for the LLM grader submission")
        logger.info(f"Submitted local LLM grader with Slurm job ID: {job_id}")
        global _GRADER_JOB_ID
        _GRADER_JOB_ID = job_id
        _ensure_grader_cleanup_hooks()
        nodes = _wait_for_slurm_nodes(job_id, timeout=timeout)
        node_candidates = _expand_slurm_node_list(nodes)
        if not node_candidates:
            raise RuntimeError(f"Unable to determine head node from Slurm node list: {nodes}")
        node = node_candidates[0]
        os.environ["OPENAI_BASE_URL"] = f"http://{node}:{vllm_port}/v1"
        os.environ["OPENAI_API_KEY"] = "grader"
        health_url = f"http://{node}:{vllm_port}/health"
        health_retries = int(os.environ.get("HEALTH_CHECK_RETRIES", "120"))
        health_delay = int(os.environ.get("HEALTH_CHECK_DELAY", "10"))
        _wait_for_vllm_health(health_url, retries=health_retries, delay=health_delay)
        logger.info(
            "LLM grader job %s scheduled on node(s): %s; OPENAI_BASE_URL=%s",
            job_id,
            nodes,
            os.environ["OPENAI_BASE_URL"],
        )
    else:
        from huggingface_hub import get_inference_endpoint, get_token

        endpoint = get_inference_endpoint(name=name, namespace=namespace)
        if endpoint.status == "running":
            logger.info(f"LLM grader endpoint {name} is already running at URL: {endpoint.url}")
        else:
            logger.info(f"Waking up Hugging Face endpoint {name}...")
            endpoint.resume()
            endpoint.wait(timeout=timeout)
            logger.info(f"LLM grader endpoint {name} is now running at URL: {endpoint.url}")
        os.environ["OPENAI_BASE_URL"] = f"{endpoint.url}/v1"
        os.environ["OPENAI_API_KEY"] = get_token()
        os.environ["HF_ENDPOINT_REPO"] = endpoint.repository
