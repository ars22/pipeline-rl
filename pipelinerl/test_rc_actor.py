"""
Standalone RC smoke harness.

This script:
1. Loads an RC config (default: conf/rc_smoke.yaml)
2. Starts RC actor and summarization vLLM servers using the same path as launch.py
3. Starts environment servers
4. Runs the RC actor loop
"""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from pipelinerl.launch import (
    _RC_ACTOR_VLLM_INTERNAL_PORT_BASE,
    _SUMMARIZATION_VLLM_INTERNAL_PORT_BASE,
    _append_vllm_kwargs,
    _apply_model_compat_overrides,
    _with_vllm_runtime_env,
)
from pipelinerl.world import Job, WorldMap

DEFAULT_RC_TEST_CONFIG = "rc_smoke"
_OUTPUT_DIR_FILE = "/tmp/test_rc_actor_output_dir.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _popen(
    cmd: list[str],
    env: dict | None = None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        env=env if env else os.environ,
        stdout=stdout,
        stderr=stderr,
        preexec_fn=os.setsid,
    )


def save_command(log_dir: Path, cmd: list[str]):
    with open(log_dir / "command.txt", "w") as f:
        f.write(" ".join(cmd))


def _get_reasoning_model(cfg: DictConfig, exp_dir: Path) -> tuple[str | Path, str | None]:
    finetune_model_path = exp_dir / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        return finetune_model_path, None
    model_path = cfg.model_path
    if model_path is None:
        raise ValueError("model_path must be defined")
    return model_path, cfg.get("model_revision")


def _get_summarization_model(cfg: DictConfig, exp_dir: Path) -> tuple[str | Path, str | None]:
    if cfg.get("summarization_model_path") is not None:
        model_path = cfg.summarization_model_path
        if model_path is None:
            raise ValueError("summarization_model_path must be defined")
        return model_path, cfg.get("summarization_model_revision")
    return _get_reasoning_model(cfg, exp_dir)


def _get_vllm_entrypoint(vllm_config: DictConfig) -> str:
    return (
        "pipelinerl.entrypoints.run_vllm1"
        if bool(getattr(vllm_config, "use_v1", False))
        else "pipelinerl.entrypoints.run_vllm0"
    )


def _start_vllm_server(
    *,
    model_path: str | Path,
    model_revision: str | None,
    vllm_config: DictConfig,
    port: int,
    seed: int,
    gpus: list[int],
    log_dir: Path,
    port_seed: int,
):
    os.makedirs(log_dir, exist_ok=True)
    cmd = [
        "python",
        "-m",
        _get_vllm_entrypoint(vllm_config),
        "--model",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--seed",
        str(seed),
        "--disable-weight-updates",
    ]
    if model_revision:
        cmd.extend(["--revision", str(model_revision)])
    _append_vllm_kwargs(cmd, bool(getattr(vllm_config, "use_v1", False)), vllm_config.vllm_kwargs)

    gpu_str = ",".join(str(gpu) for gpu in gpus)
    logger.info("Starting vLLM server with command: %s on gpus: %s", " ".join(cmd), gpu_str)
    save_command(log_dir, cmd)

    log_file_path = log_dir / "stdout.log"
    err_file_path = log_dir / "stderr.log"
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        return _popen(
            cmd,
            env=_with_vllm_runtime_env(gpu_str, port_seed),
            stdout=log_file,
            stderr=err_file,
        )


def start_rc_actor_llm(cfg: DictConfig, job: Job, exp_dir: Path):
    model_path, model_revision = _get_reasoning_model(cfg, exp_dir)
    vllm_config = cfg.get("rc_actor_vllm_config") or cfg.vllm_config
    if job.port is None:
        raise ValueError("RC actor LLM job is missing its port")
    return _start_vllm_server(
        model_path=model_path,
        model_revision=model_revision,
        vllm_config=vllm_config,
        port=job.port,
        seed=cfg.seed + job.replica_idx,
        gpus=job.gpus,
        log_dir=exp_dir / f"rc_actor_vllm_{job.replica_idx}",
        port_seed=_RC_ACTOR_VLLM_INTERNAL_PORT_BASE + job.local_idx * 100,
    )


def start_summarization_llm(cfg: DictConfig, job: Job, exp_dir: Path):
    model_path, model_revision = _get_summarization_model(cfg, exp_dir)
    vllm_config = cfg.get("summarization_vllm_config") or cfg.vllm_config
    if job.port is None:
        raise ValueError("Summarization LLM job is missing its port")
    return _start_vllm_server(
        model_path=model_path,
        model_revision=model_revision,
        vllm_config=vllm_config,
        port=job.port,
        seed=cfg.seed + 1000 + job.replica_idx,
        gpus=job.gpus,
        log_dir=exp_dir / f"summarization_vllm_{job.replica_idx}",
        port_seed=_SUMMARIZATION_VLLM_INTERNAL_PORT_BASE + job.local_idx * 100,
    )


def wait_for_llm_server(port: int, timeout: int = 300) -> bool:
    import requests

    start_time = time.time()
    url = f"http://localhost:{port}/v1/models"
    logger.info("Waiting for LLM server on port %s to be ready...", port)
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("LLM server on port %s is ready!", port)
                return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("LLM server on port %s did not become ready in %ss", port, timeout)
    return False


def start_environment(cfg: DictConfig, job: Job, exp_dir: Path):
    if job.port is None:
        raise ValueError("Environment job is missing its port")

    run_dir = exp_dir / f"environment_{job.replica_idx}"
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_environment",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={run_dir}",
        f"me.job_idx={job.idx}",
    ]
    logger.info("Starting environment %s on port %s", job.replica_idx, job.port)
    save_command(run_dir, cmd)

    log_file_path = run_dir / "stdout.log"
    err_file_path = run_dir / "stderr.log"
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        return _popen(
            cmd,
            env=dict(os.environ),
            stdout=log_file,
            stderr=err_file,
        )


def wait_for_environment(port: int, timeout: int = 60) -> bool:
    import requests

    start_time = time.time()
    url = f"http://localhost:{port}/health"
    logger.info("Waiting for environment server on port %s to be ready...", port)
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("Environment server on port %s is ready!", port)
                return True
        except Exception:
            pass
        time.sleep(3)
    logger.error("Environment server on port %s did not become ready in %ss", port, timeout)
    return False


def prepare_config_for_test(cfg: DictConfig, output_dir: Path) -> tuple[DictConfig, WorldMap]:
    OmegaConf.set_struct(cfg, False)
    cfg.output_dir = str(output_dir)
    if not OmegaConf.select(cfg, "me"):
        OmegaConf.update(cfg, "me", {})
    _apply_model_compat_overrides(cfg)
    world_map = WorldMap(cfg, verbose=False)
    cfg.jobs = [job.model_dump() for job in world_map.get_all_jobs()]
    cfg.me.llm_urls = "+".join(world_map.get_rc_actor_urls())
    summarization_urls = world_map.get_summarization_urls()
    if summarization_urls:
        cfg.me.summarization_llm_urls = "+".join(summarization_urls)
    OmegaConf.set_struct(cfg, True)
    return cfg, world_map


def _get_output_dir(cfg: DictConfig, my_rank: int) -> Path:
    if my_rank == 0:
        output_dir = Path(f"{cfg.output_dir}_{int(time.time())}")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(_OUTPUT_DIR_FILE, "w") as f:
            f.write(str(output_dir))
        return output_dir

    for _ in range(30):
        if os.path.exists(_OUTPUT_DIR_FILE):
            with open(_OUTPUT_DIR_FILE, "r") as f:
                return Path(f.read().strip())
        time.sleep(1)
    raise RuntimeError("Timeout waiting for output directory from rank 0")


@hydra.main(config_path="../conf", config_name=DEFAULT_RC_TEST_CONFIG, version_base="1.3.2")
def main(cfg: DictConfig):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    my_rank = int(os.environ.get("RANK", 0))

    if "test_world" in cfg and cfg.test_world:
        logger.warning("`test_world` is deprecated; using `world` for RC smoke topology.")

    if world_size > 1:
        all_addr = os.environ.get("ALL_ADDR", "")
        if not all_addr:
            raise ValueError("ALL_ADDR environment variable must be set when WORLD_SIZE > 1")
        nodelist = [x.strip() for x in all_addr.strip().split(",")]
        if len(nodelist) != world_size:
            raise ValueError(f"ALL_ADDR length {len(nodelist)} does not match WORLD_SIZE {world_size}")
    else:
        nodelist = ["localhost"]

    logger.info("Multi-node setup: world_size=%s my_rank=%s", world_size, my_rank)
    logger.info("Node list: %s", nodelist)

    output_dir = _get_output_dir(cfg, my_rank)
    cfg, world_map = prepare_config_for_test(cfg, output_dir)

    logger.info("=" * 80)
    logger.info("RC Actor Test Configuration (Rank %s/%s)", my_rank, world_size)
    logger.info("Model: %s", cfg.model_path)
    logger.info("Summarization model: %s", cfg.get("summarization_model_path", cfg.model_path))
    logger.info("Output directory: %s", output_dir)
    logger.info("Jobs on this rank: %s", world_map.get_jobs_on_rank(my_rank))
    logger.info("=" * 80)

    config_dir = output_dir / "conf"
    os.makedirs(config_dir, exist_ok=True)
    with open(config_dir / "exp_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    processes: list[subprocess.Popen] = []
    output_dir_for_log = output_dir

    try:
        if my_rank == 0:
            env_jobs = [job for job in world_map.get_jobs_on_rank(my_rank) if job.kind == "environment"]
            if env_jobs:
                logger.info("=" * 80)
                logger.info("Step 1: Starting environment servers")
                logger.info("=" * 80)
            for job in env_jobs:
                processes.append(start_environment(cfg, job, output_dir))
            for job in env_jobs:
                assert job.port is not None
                if not wait_for_environment(job.port, timeout=60):
                    logger.error("Environment server on port %s failed to start!", job.port)
                    return 1

        rank_jobs = world_map.get_jobs_on_rank(my_rank)
        rc_actor_jobs = [job for job in rank_jobs if job.kind == "rc_actor_llm"]
        if rc_actor_jobs:
            logger.info("=" * 80)
            logger.info("Step 2: Starting RC actor vLLM servers on node %s", my_rank)
            logger.info("=" * 80)
        for job in rc_actor_jobs:
            logger.info("Starting RC actor LLM %s on GPUs %s port %s", job.replica_idx, job.gpus, job.port)
            processes.append(start_rc_actor_llm(cfg, job, output_dir))
        for job in rc_actor_jobs:
            assert job.port is not None
            if not wait_for_llm_server(job.port, timeout=300):
                logger.error("RC actor LLM server on port %s failed to start!", job.port)
                return 1

        summarization_jobs = [job for job in rank_jobs if job.kind == "summarization_llm"]
        if summarization_jobs:
            logger.info("=" * 80)
            logger.info("Step 3: Starting summarization vLLM servers on node %s", my_rank)
            logger.info("=" * 80)
        for job in summarization_jobs:
            logger.info("Starting summarization LLM %s on GPUs %s port %s", job.replica_idx, job.gpus, job.port)
            processes.append(start_summarization_llm(cfg, job, output_dir))
        for job in summarization_jobs:
            assert job.port is not None
            if not wait_for_llm_server(job.port, timeout=300):
                logger.error("Summarization LLM server on port %s failed to start!", job.port)
                return 1

        if my_rank == 0:
            logger.info("=" * 80)
            logger.info("Step 4: Running RC actor")
            logger.info("=" * 80)
            from pipelinerl import rc_actor

            logger.info("Starting RC actor loop...")
            logger.info("Output directory: %s", output_dir)
            logger.info("RC stream path: %s", output_dir / "streams" / "rc_actor")
            rc_actor.run_actor_loop(cfg)
            return 0

        logger.info("=" * 80)
        logger.info("Node %s: inference servers running, waiting for termination signal...", my_rank)
        logger.info("=" * 80)
        try:
            signal.pause()
        except KeyboardInterrupt:
            logger.info("Node %s: received termination signal", my_rank)
        return 0
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error("Test failed with exception: %s", e, exc_info=True)
        return 1
    finally:
        logger.info("=" * 80)
        logger.info("Cleanup: Stopping all servers")
        logger.info("=" * 80)
        for process in processes:
            if process:
                try:
                    logger.info("Stopping process %s", process.pid)
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except Exception as e:
                    logger.warning("Error stopping process: %s", e)
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception:
                        pass
        logger.info("Test artifacts saved to: %s", output_dir_for_log)


if __name__ == "__main__":
    main()
