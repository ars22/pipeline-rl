import logging
import math
import os
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import List, TextIO

import hydra
from omegaconf import DictConfig, OmegaConf

from pipelinerl.grader_launch import start_llm_grader
from pipelinerl.state import TrainerState
from pipelinerl.streams import SingleStreamSpec, connect_to_redis, read_stream, set_streams_backend, write_to_streams
from pipelinerl.utils import terminate_with_children
from pipelinerl.world import Job, WorldMap

logger = logging.getLogger(__name__)

# All the launch commands in this file pass the environment to child processes
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["TORCH_DISABLE_SHARE_RDZV_TCP_STORE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

_V1_UNSUPPORTED_VLLM_KWARGS = {"disable-log-requests", "num-scheduler-steps"}
_ACTOR_VLLM_INTERNAL_PORT_BASE = 38000
_RC_ACTOR_VLLM_INTERNAL_PORT_BASE = 39000
_REF_VLLM_INTERNAL_PORT_BASE = 40000
_SUMMARIZATION_VLLM_INTERNAL_PORT_BASE = 41000

def _popen(
    cmd: list[str],
    env: dict | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> subprocess.Popen:
    """Wrapper around subprocess.Popen that allows for easier debugging."""
    if os.environ.get("DRY_RUN", "0") == "1":
        return  # type: ignore
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr,
    )


def _append_vllm_kwargs(cmd: list[str], use_v1: bool, kwargs: dict | None) -> None:
    if not kwargs:
        return

    filtered_kwargs = dict(kwargs)
    if use_v1:
        for key in sorted(_V1_UNSUPPORTED_VLLM_KWARGS):
            if key in filtered_kwargs:
                logger.info("Ignoring V0-only vLLM kwarg on V1 path: --%s", key)
                filtered_kwargs.pop(key, None)

    for key, value in filtered_kwargs.items():
        cmd.append(f"--{key}")
        if value not in [None, ""]:
            cmd.append(str(value))


def _with_vllm_runtime_env(gpu_str: str, port_seed: int | None = None) -> dict[str, str]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str}
    if port_seed is not None:
        env["VLLM_PORT"] = str(port_seed)
    return env


def _is_finetune_process(proc: subprocess.Popen) -> bool:
    args = proc.args
    if isinstance(args, (list, tuple)):
        command = " ".join(str(part) for part in args)
    else:
        command = str(args)
    return "run_finetune.py" in command


def validate_config(cfg: DictConfig):
    if cfg.world.finetune_fraction == 0:
        return
    if cfg.world.preprocessor_fraction == 0 and cfg.finetune.rl.kl_coef > 0.0:
        raise ValueError("Preprocessor fraction must be > 0 if KL is used")
    
    # Check for vision language model constraints
    if cfg.finetune.model_class == "vision2seq-language-modeling":
        model_id = cfg.model_path
        if not model_id or "Qwen2.5-VL" not in model_id:
            raise ValueError("Only Qwen2.5-VL models are supported for vision language modeling")
        if cfg.finetune.seq_packing:
            raise ValueError("Vision language models cannot use sequence packing (seq_packing must be false)")
        if cfg.finetune.train_batch_size > 1:
            raise ValueError("Vision language models cannot use batch size > 1 (train_batch_size must be 1)")
    
    if cfg.finetune.seq_parallel > 1:
        if not cfg.finetune.seq_packing:
            raise ValueError("seq_parallel > 1 requires seq_packing to be true")
    
    if cfg.preprocess.dataset_buffer_size > 0:
        if cfg.preprocess.dataset_buffer_size != cfg.preprocess.ring_buffer_size:
            raise ValueError("dataset_buffer_size must be equal to ring_buffer_size")
        if cfg.pop_old_data:
            raise ValueError("Cannot use pop_old_data with preprocessor dataset_buffer_size > 0")

    # Check for value loss coefficient constraints
    if cfg.finetune.model_class == "causal-language-modeling-with-value-head":
        if not hasattr(cfg.finetune.rl, "value_loss_coef") or cfg.finetune.rl.value_loss_coef <= 0.0:
            raise ValueError("value_loss_coef must be greater than 0 when using causal-language-modeling-with-value-head")


def run_ref_llm(cfg: DictConfig, preprocessor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path):
    # Use actor_vllm_config if available, else fall back to vllm_config
    actor_vllm_cfg = cfg.get("actor_vllm_config")
    if actor_vllm_cfg is None:
        actor_vllm_cfg = cfg.vllm_config
    kwargs = actor_vllm_cfg.vllm_kwargs
    if kwargs.get("num-scheduler-steps", 1) > 1:
        kwargs["num-scheduler-steps"] = 1
        logger.warning("Set num-scheduler-steps to 1 for reference vLLM")
    log_dir = exp_dir / f"ref_vllm_{preprocessor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)

    model_id = cfg.model_path
    model_revision = cfg.get("model_revision")
    if model_id is None:
        raise ValueError("model_path must be defined")
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_id),
        "--port",
        str(8180 + local_idx),
        "--host",
        "0.0.0.0",
        "--seed",
        str(cfg.seed + preprocessor_llm_idx),
    ]
    if model_revision:
        cmd.extend(["--revision", str(model_revision)])

    # Add vLLM kwargs as separate arguments
    _append_vllm_kwargs(cmd, bool(getattr(actor_vllm_cfg, "use_v1", False)), kwargs)

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running reference LLM with command: {' '.join(cmd)} with gpus: {gpu_str}")
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env=_with_vllm_runtime_env(
                gpu_str,
                _REF_VLLM_INTERNAL_PORT_BASE + local_idx * 100,
            ),
            stdout=log_file,
            stderr=err_file,
        )


def run_summarization_llm(
    cfg: DictConfig, summarization_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path
):
    # Use summarization model path if specified, otherwise use main model
    if cfg.get("summarization_model_path") is not None:
        model_id = cfg.summarization_model_path
        model_revision = cfg.get("summarization_model_revision")
        if model_id is None:
            raise ValueError("summarization_model_path must be defined")
    else:
        model_id = cfg.model_path
        model_revision = cfg.get("model_revision")
        if model_id is None:
            raise ValueError("model_path must be defined")
    
    # Use summarization vllm config if specified, otherwise use main vllm config
    vllm_cfg = cfg.get("summarization_vllm_config")
    if vllm_cfg is None:
        vllm_cfg = cfg.vllm_config
    kwargs = vllm_cfg.vllm_kwargs.copy() if vllm_cfg.vllm_kwargs else {}
    
    log_dir = exp_dir / f"summarization_vllm_{summarization_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)

    # Use custom entrypoint (run_vllm0/run_vllm1) to support newer models like Qwen3
    # Same logic as actor servers, but without weight update parameters
    entrypoint = (
        "pipelinerl.entrypoints.run_vllm1" 
        if vllm_cfg.use_v1 else 
        "pipelinerl.entrypoints.run_vllm0"
    )
    cmd = [
        "python",
        "-m",
        entrypoint,
        "--model",
        str(model_id),
        "--port",
        str(8200 + local_idx),  # Use 8200+ for summarization LLMs
        "--host",
        "0.0.0.0",
        "--seed",
        str(cfg.seed + summarization_llm_idx + 1000),
        "--disable-weight-updates",  # Summarization servers don't need weight updates
    ]
    if model_revision:
        cmd.extend(["--revision", str(model_revision)])

    # Add vLLM kwargs as separate arguments
    _append_vllm_kwargs(cmd, bool(vllm_cfg.use_v1), kwargs)

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running summarization LLM with command: {' '.join(cmd)} with gpus: {gpu_str}")
    save_command(log_dir, cmd)
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env=_with_vllm_runtime_env(
                gpu_str,
                _SUMMARIZATION_VLLM_INTERNAL_PORT_BASE + local_idx * 100,
            ),
            stdout=log_file,
            stderr=err_file,
        )


def run_actor_llm(
    cfg: DictConfig, world_map: WorldMap, actor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path
):
    finetune_model_path = exp_dir / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
        actor_model_revision = None
    else:
        actor_model_path = cfg.model_path
        actor_model_revision = cfg.get("model_revision")
        if actor_model_path is None:
            raise ValueError("model_path must be defined")

    # Use actor_vllm_config if specified, otherwise use main vllm_config
    vllm_cfg = cfg.get("actor_vllm_config")
    if vllm_cfg is None:
        vllm_cfg = cfg.vllm_config

    # TODO: add support for tensor and process parallelism
    log_dir = exp_dir / f"actor_vllm_{actor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    entrypoint = (
        "pipelinerl.entrypoints.run_vllm1" 
        if vllm_cfg.use_v1 else 
        "pipelinerl.entrypoints.run_vllm0"
    )
    cmd = [
        "python",
        "-m",
        entrypoint,
        "--model",
        str(actor_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8080 + local_idx),
        "--seed",
        str(cfg.seed + actor_llm_idx),
        "--actor-llm-idx",
        str(actor_llm_idx),
        "--weight-update-group-init-method",
        f"tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        "--weight-update-group-world-size",
        str(world_map.weight_update_group_size),
    ]
    if actor_model_revision:
        cmd.extend(["--revision", str(actor_model_revision)])

    # Add vLLM kwargs as separate arguments
    _append_vllm_kwargs(cmd, bool(vllm_cfg.use_v1), vllm_cfg.vllm_kwargs)

    # Disable weight updates in debug mode or eval_only mode
    if cfg.debug.mode or cfg.get('eval_only', False):
        cmd.append("--disable-weight-updates")

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running actor_llm with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env=_with_vllm_runtime_env(
                gpu_str,
                _ACTOR_VLLM_INTERNAL_PORT_BASE + local_idx * 100,
            ),
            stdout=log_file,
            stderr=err_file,
        )


def run_rc_actor_llm(
    cfg: DictConfig, world_map: WorldMap, rc_actor_llm_idx: int, local_idx: int, gpus: list[int], exp_dir: Path
):
    finetune_model_path = exp_dir / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        rc_actor_model_path = finetune_model_path
        rc_actor_model_revision = None
    else:
        rc_actor_model_path = cfg.model_path
        rc_actor_model_revision = cfg.get("model_revision")
        if rc_actor_model_path is None:
            raise ValueError("model_path must be defined")

    log_dir = exp_dir / f"rc_actor_vllm_{rc_actor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use rc_actor_vllm_config if available, else fall back to vllm_config
    rc_actor_vllm_cfg = cfg.get("rc_actor_vllm_config")
    if rc_actor_vllm_cfg is None:
        rc_actor_vllm_cfg = cfg.vllm_config
    
    entrypoint = (
        "pipelinerl.entrypoints.run_vllm1" 
        if rc_actor_vllm_cfg.use_v1 else 
        "pipelinerl.entrypoints.run_vllm0"
    )
    cmd = [
        "python",
        "-m",
        entrypoint,
        "--model",
        str(rc_actor_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8000 + local_idx),  # Use 8000+ for RC actor LLMs
        "--seed",
        str(cfg.seed + rc_actor_llm_idx),
        "--actor-llm-idx",
        str(rc_actor_llm_idx),
        "--weight-update-group-init-method",
        f"tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        "--weight-update-group-world-size",
        str(world_map.weight_update_group_size),
    ]
    if rc_actor_model_revision:
        cmd.extend(["--revision", str(rc_actor_model_revision)])

    # Add vLLM kwargs as separate arguments
    _append_vllm_kwargs(cmd, bool(rc_actor_vllm_cfg.use_v1), rc_actor_vllm_cfg.vllm_kwargs)

    # Disable weight updates in debug mode or eval_only mode
    if cfg.debug.mode or cfg.get('eval_only', False):
        cmd.append("--disable-weight-updates")

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Running rc_actor_llm with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env=_with_vllm_runtime_env(
                gpu_str,
                _RC_ACTOR_VLLM_INTERNAL_PORT_BASE + local_idx * 100,
            ),
            stdout=log_file,
            stderr=err_file,
        )


def run_rc_actor(world_map: WorldMap, rc_actor_idx: int, exp_dir: Path):
    if rc_actor_idx != 0:
        raise NotImplementedError("Can only do 1 rc_actor yet")
    llm_urls = "+".join(world_map.get_rc_actor_urls())
    summarization_llm_urls = world_map.get_summarization_urls()
    
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_rc_actor",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/rc_actor",
        f"+me.llm_urls={llm_urls}",
    ]
    
    # Add summarization LLM URLs if they exist
    if summarization_llm_urls:
        cmd.append(f"+me.summarization_llm_urls={'+'.join(summarization_llm_urls)}")
    
    logger.info(f"Running RC actor with command: {' '.join(cmd)}")
    save_command(exp_dir / "rc_actor", cmd)
    yield _popen(
        cmd,
        env=dict(os.environ),
    )


def run_actor(world_map: WorldMap, actor_idx: int, exp_dir: Path):
    if actor_idx != 0:
        raise NotImplementedError("Can only do 1 actor yet")
    llm_urls = "+".join(world_map.get_actor_urls())
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_actor",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/actor",
        f"+me.llm_urls={llm_urls}",
    ]
    logger.info(f"Running actor with command: {' '.join(cmd)}")
    save_command(exp_dir / "actor", cmd)
    yield _popen(
        cmd,
        env=dict(os.environ),
    )


def run_environment(cfg: DictConfig, job: Job):
    # run in a subprocess like in the rest of the code
    run_dir = Path(cfg.output_dir) / f"environment_{job.replica_idx}"
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_environment",
        "--config-dir",
        f"{cfg.output_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={cfg.output_dir}",
        f"hydra.run.dir={str(run_dir)}",
        f"me.job_idx={job.idx}",
    ]
    logger.info(f"Running environment with command: {' '.join(cmd)}")
    os.makedirs(run_dir, exist_ok=True)    
    save_command(run_dir, cmd)
    log_file_path = str(run_dir / "stdout.log")
    err_file_path = str(run_dir / "stderr.log")
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        yield _popen(
            cmd,
            env=dict(os.environ),
            stdout=log_file,
            stderr=err_file,
        )


def run_finetune(cfg: DictConfig, world_map: WorldMap, gpus: list[int], exp_dir: Path):
    if cfg.use_fsdp and cfg.use_deepspeed:
        raise ValueError("Cannot use both FSDP and DeepSpeed")
    cmd = [
        "python",
        "-m",
        "accelerate.commands.launch",
    ]
    if world_map.world_size > 1:
        # DeepSpeed multi-node args
        assert cfg.use_deepspeed
        # assert world_map.master_addr.startswith("dns-") and world_map.master_addr.endswith("-0")
        # hosts = [world_map.master_addr[:-2] + f"-{i}" for i in range(world_map.world_size)]
        
        hosts = world_map.nodelist
        
        filter_parts = []
        for rank, job_list in world_map.job_map.items():
            for job in job_list:
                if job.kind == "finetune":
                    filter_parts.append(f"{hosts[rank]}:{','.join(map(str, job.gpus))}")
        deepspeed_include_filter = "@".join(filter_parts)
        logger.info(f"Deepspeed include filter: {deepspeed_include_filter}")
        # Orchestrator rank must have already created hostfile.txt
        hostfile_path = str(exp_dir / "hostfile.txt")
        cmd += [
            "--num_machines",
            str(len(world_map.nodes_with_finetuning())),
            "--machine_rank",
            str(world_map.my_finetuning_rank()),
            "--main_process_ip",
            str(os.environ.get("MASTER_ADDR")),
            "--main_process_port",
            str(os.environ.get("MASTER_PORT")),
            "--deepspeed_hostfile",
            hostfile_path,
            "--deepspeed_inclusion_filter",
            deepspeed_include_filter,
            "--deepspeed_multinode_launcher",
            "nossh"
        ]
    # get path to this file
    this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    if cfg.use_deepspeed:
        # DeepSpeed single-node args
        cmd += [
            "--use_deepspeed",
            "--deepspeed_config_file",
            str(this_file_path / f"../conf/deepspeed/{cfg.deepspeed_config}.json"),
        ]
    # DeepSpeed and non-DeepSpeed args
    accelerate_config = cfg.accelerate_config
    if accelerate_config is None:
        if cfg.use_deepspeed:
            accelerate_config = "deepspeed"
        elif cfg.use_fsdp:
            accelerate_config = "fsdp_mp"
        else:
            accelerate_config = "base_mp"
    cmd += [
        "--config_file",
        str(this_file_path / f"../conf/accelerate/{accelerate_config}.yaml"),
        "--rdzv_backend",
        "c10d",
    ]
    if gpus:
        gpus_str = str(",".join([str(gpu) for gpu in gpus])) if len(gpus) < world_map.node_size else "all"
        cmd += [
            "--gpu-ids",
            gpus_str,
        ]
    cmd += [
        "--num_processes",
        str(world_map.total_finetune_gpus),
        "pipelinerl/entrypoints/run_finetune.py",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/finetune",
        # TODO: figure out why we can't build WorldMap in run_finetune.py
        # Current workaround: pass the essential information as follows:
        f"+me.weight_update_group_init_method=tcp://{world_map.master_addr}:{cfg.world.actor_group_port}",
        f"+me.weight_update_group_world_size={world_map.weight_update_group_size}",
        f"+me.llm_urls={'+'.join(world_map.get_rc_actor_urls() + world_map.get_actor_urls())}",
    ]
    if cfg.debug.mode in ["finetune", "open_loop", "finetune+preprocessor"]:
        cmd.append("finetune.send_weight_updates=False")

    logger.info(f"Running finetune with command: {' '.join(cmd)}")
    save_command(exp_dir / "finetune", cmd)
    env = dict(os.environ)
    env["DS_ENV_FILE"] = str(exp_dir / ".deepspeed_env")
    yield _popen(cmd, env=env)


def run_preprocess(world_map: WorldMap, preprocessor_idx: int, exp_dir: Path):
    if preprocessor_idx != 0:
        raise NotImplementedError("Can only do 1 preprocessor yet")
    llm_urls = "+".join(world_map.get_preprocessor_urls())
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_preprocess",
        "--config-dir",
        f"{exp_dir}/conf",
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={exp_dir}/preprocess",
        f"+me.llm_urls={llm_urls}",
    ]
    logger.info(f"Running preprocess with command: {' '.join(cmd)}")
    save_command(exp_dir / "preprocess", cmd)
    yield _popen(
        cmd,
        env=dict(os.environ),
    )


def run_redis(cfg: DictConfig):
    # Launch redis-server
    cmd = [
        "redis-server",
        "--bind",
        "0.0.0.0",
        "--port",
        str(cfg.streams.port),
        "--dir",
        str(cfg.output_dir),
        "--protected-mode",
        "no",
        "--save",
        cfg.streams.save,
    ]
    logger.info(f"Running redis with command: {' '.join(cmd)}")
    save_command(Path(cfg.output_dir) / "redis", cmd)
    yield _popen(cmd, env=dict(os.environ))


def save_command(script_dir: Path, cmd):
    os.makedirs(script_dir, exist_ok=True)
    script_path = script_dir / "start.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        # Properly quote arguments for the shell script
        quoted_cmd = [f"'{arg}'" if " " in arg or "$" in arg else arg for arg in cmd]
        f.write(" ".join(quoted_cmd) + "\n")
    os.chmod(script_path, 0o755)
    logger.info(f"Saved start script to {script_path}")


def clean_up(exp_dir, force_restart):
    logger.info("Cleaning up streams directory")
    if os.path.exists(f"{exp_dir}/streams"):
        if os.path.isdir(f"{exp_dir}/streams") and not os.path.islink(f"{exp_dir}/streams"):
            shutil.rmtree(f"{exp_dir}/streams")
        else:
            os.remove(f"{exp_dir}/streams")
    if os.path.exists(f"{exp_dir}/dump.rdb"):
        os.remove(f"{exp_dir}/dump.rdb")

    if force_restart:
        if os.path.exists(f"{exp_dir}/finetune"):
            logger.info("Cleaning up finetune directory")
            shutil.rmtree(f"{exp_dir}/finetune")

        # erase all the logs
        log_files = list(exp_dir.glob("**/*.log"))
        for log_file in log_files:
            logger.info(f"Erasing {log_file}")
            with open(log_file, "r"):
                pass


def watch_processes_running(exp_path: Path, processes: List[subprocess.Popen], debug_mode: bool = False):
    if not debug_mode:
        trainer_state = TrainerState(exp_path)
        trainer_state.start_listening()
    else:
        trainer_state = None

    # Wait for all processes to complete
    def gently_stop_all_processes(skip_pids: set[int] | None = None):
        logger.info("\nShutting down processes...")
        # Terminate all running processes
        skip_pids = skip_pids or set()
        for proc in processes:
            if proc.pid in skip_pids:
                continue
            logger.info(f"Terminating {proc.args}")
            terminate_with_children(proc.pid)

    logger.info("I have launched everyone, waiting for them to finish...")

    # last_trainer_version = -1
    # last_time_new_version = time.time()

    try:
        # Wait for all processes to complete
        # if just one dies, stop all
        while True:
            for proc in processes:
                if (return_code := proc.poll()) is not None:
                    if return_code == 0 and _is_finetune_process(proc):
                        logger.info("Finetune process exited cleanly; shutting down remaining processes.")
                        gently_stop_all_processes(skip_pids={proc.pid})
                        return
                    logger.error(f"Process {proc.args} terminated with code {proc.returncode}")
                    gently_stop_all_processes(skip_pids={proc.pid})
                    sys.exit(1)
            # TODO: make the watcdog code below more stable
            # if (trainer_state is not None
            #     and (version := trainer_state.propagated_weight_version is not None)
            #     and version > last_trainer_version):
            #     last_trainer_version = version
            #     last_time_new_version = time.time()
            # if not debug_mode and time.time() - last_time_new_version > 1800:
            #     logger.error("No new weight update in 30 minutes, exiting")
            #     sys.exit(1)
            time.sleep(1.0)
    except KeyboardInterrupt:
        gently_stop_all_processes()


def debug_link_streams(cfg: DictConfig, topics: list[str]):
    if not cfg.debug.streams_from:
        raise ValueError("Need to specify streams_from for debug mode")
    stream_dir = Path(cfg.output_dir) / "streams"
    for topic in topics:
        source_topic_dir = Path(cfg.debug.streams_from) / "streams" / topic
        target_topic_dir = stream_dir / topic
        if not os.path.exists(source_topic_dir):
            raise ValueError(f"Source topic {source_topic_dir} does not exist")
        os.symlink(source_topic_dir, target_topic_dir)
        logger.info(f"Linked {source_topic_dir} to {target_topic_dir}")


def launch_jobs(cfg: DictConfig, world_map: WorldMap, job_kind_filter: list | None = None):
    exp_dir = Path(cfg.output_dir)
    processes = []
    all_job_kinds = ["rc_actor", "rc_actor_llm", "summarization_llm", "actor", "environment", "actor_llm", "preprocessor", "preprocessor_llm", "finetune"]
    # for rank in range(world_map.world_size):
    logger.info(f"Jobs on rank {world_map.my_rank}: {world_map.get_jobs_on_rank(world_map.my_rank)}")
    if job_kind_filter is None:
        job_kind_filter = all_job_kinds
    for job in world_map.get_jobs_on_rank(world_map.my_rank):
        if job.kind not in all_job_kinds:
            raise ValueError(f"Unknown job kind {job.kind}")
        if job.kind not in job_kind_filter:
            continue
        if job.kind == "rc_actor":
            processes.extend(run_rc_actor(world_map, job.replica_idx, exp_dir))
        elif job.kind == "rc_actor_llm":
            if cfg.debug.use_existing_llms:
                continue
            processes.extend(run_rc_actor_llm(cfg, world_map, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "summarization_llm":
            if cfg.debug.use_existing_llms:
                continue
            processes.extend(run_summarization_llm(cfg, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "actor":
            processes.extend(run_actor(world_map, job.replica_idx, exp_dir))
        elif job.kind == "environment":
            processes.extend(run_environment(cfg, job))
        elif job.kind == "actor_llm":
            if cfg.debug.use_existing_llms:
                continue
            processes.extend(run_actor_llm(cfg, world_map, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "preprocessor":
            processes.extend(run_preprocess(world_map, job.replica_idx, exp_dir))
        elif job.kind == "preprocessor_llm":
            if cfg.debug.use_existing_llms:
                continue            
            processes.extend(run_ref_llm(cfg, job.replica_idx, job.local_idx, job.gpus, exp_dir))
        elif job.kind == "finetune":
            processes.extend(run_finetune(cfg, world_map, job.gpus, exp_dir))
        else:
            raise ValueError(f"Unknown job kind {job.kind}")
    
    return processes


def setup_logging(log_file: Path):
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger.info("Logging setup complete")


@hydra.main(
    config_path="../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    # Resolve all interpolations in config (e.g., ${actor.problem_queue_size} // ${world.actor_fraction})
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Log the resolved config with all computed values
    logger.info("="*80)
    logger.info("Configuration (with resolved values):")
    logger.info("="*80)
    # Print config line by line for better formatting in logs
    config_yaml = OmegaConf.to_yaml(resolved_cfg)
    for line in config_yaml.split('\n'):
        if line.strip():  # Only print non-empty lines
            logger.info(line)
    logger.info("="*80)
    
    validate_config(cfg)

    rank = int(os.environ.get("RANK", "0"))

    # Spin up LLM grader if specified
    if "local" in cfg.llm_grader and not cfg.llm_grader.local:
        logger.info(f"LLM grader is not local, skipping launch")
    else:
        if rank == 0:
            start_llm_grader(
                cfg.llm_grader.name,
                vllm_kwargs=getattr(cfg.llm_grader, "vllm_kwargs", None),
            )
        else:
            logger.info(
                "Skipping LLM grader launch on rank %s; waiting for master to provision it",
                rank,
            )

    exp_dir = Path(cfg.output_dir)
    config_dir = exp_dir / "conf"

    os.makedirs(exp_dir / "launcher", exist_ok=True)
    log_file = exp_dir / "launcher" / f"launcher_{rank}.log"
    setup_logging(log_file)
    world_map = WorldMap(cfg, verbose=True)
    cfg.jobs = [job.model_dump() for job in world_map.get_all_jobs()]

    group = str(exp_dir)
    root = cfg.wandb.wandb_workspace_root
    if root:
        if not group.startswith(root + "/"):
            raise ValueError(f"run_dir {exp_dir} does not start with root {root}")
        cfg.wandb.wandb_group = group[len(root) + 1 :]
    if world_map.total_finetune_gpus:
        accum_passes = cfg.finetune.gradient_accumulation_passes
        n_gpus = world_map.total_finetune_gpus
        if accum_passes % n_gpus != 0:
            new_accum_passes = math.ceil(accum_passes / n_gpus) * n_gpus
            logger.warning(
                f"Adjusting gradient_accumulation_passes from {accum_passes} to {new_accum_passes} "
                f"to make it divisible by {n_gpus} processes"
            )
            cfg.finetune.gradient_accumulation_passes = new_accum_passes
    if cfg.streams.backend == "redis":
        cfg.streams.host = world_map.master_addr
    set_streams_backend(**cfg.streams)

    processes = []

    lead_launcher_stream = SingleStreamSpec(exp_path=exp_dir, topic="launcher_0")
    init_msg = {"exp_init": "true"}
    if world_map.my_rank == 0:
        clean_up(exp_dir, cfg.force_restart)
        os.makedirs(config_dir, exist_ok=True)
        OmegaConf.save(cfg, config_dir / "exp_config.yaml")
        logger.info("Orchestrator 0 created the exp folder")
        
        # Ensure streams directory exists for file-based streams
        if cfg.streams.backend == "files":
            streams_dir = exp_dir / "streams"
            os.makedirs(streams_dir, exist_ok=True)
            logger.info(f"Created streams directory at {streams_dir}")
        
        if cfg.streams.backend == "redis":
            processes.extend(run_redis(cfg))
            redis = connect_to_redis(cfg.streams)
            redis.flushall()

        if world_map.world_size > 1:
            # assert world_map.master_addr.startswith("dns-") and world_map.master_addr.endswith("-0")
            # hosts = [world_map.master_addr[:-2] + f"-{i}" for i in range(world_map.world_size)]
            
            hosts = world_map.nodelist
        
            hostfile_lines = [f"{host} slots=8" for host in hosts]
            deepspeed_hostfile_content = "\n".join(hostfile_lines)
            hostfile_path = str(exp_dir / "hostfile.txt")
            with open(hostfile_path, "w") as f:
                f.write(deepspeed_hostfile_content)
            logger.info(f"Deepspeed hostfile content:\n{deepspeed_hostfile_content}")
            logger.info(f"Orchestrator 0 created hostfile at {hostfile_path}")

        with write_to_streams(lead_launcher_stream) as stream:
            stream.write(init_msg)
        
        # Give filesystem time to sync (especially important for NFS/shared filesystems)
        if world_map.world_size > 1:
            import time
            time.sleep(0.5)
            logger.info("Rank 0 finished writing launcher stream, waiting for filesystem sync")
        
        if cfg.debug.mode == "finetune":
            debug_link_streams(cfg, [cfg.finetune.input])
        elif cfg.debug.mode == "preprocessor":
            debug_link_streams(cfg, [cfg.preprocess.input])
        elif cfg.debug.mode == "finetune+preprocessor":
            debug_link_streams(cfg, [cfg.preprocess.input])
    else:
        with read_stream(lead_launcher_stream) as stream:
            if (msg := next(stream.read())) != init_msg:
                raise ValueError(f"Expected {init_msg}, got {msg}")
        logger.info(f"Orchestrator {world_map.my_rank} heard that the exp folder is ready.")

    if cfg.debug.mode == "finetune":
        processes.extend(launch_jobs(cfg, world_map, ["finetune"]))
    elif cfg.debug.mode == "actor":
        processes.extend(launch_jobs(cfg, world_map, ["actor", "environment", "actor_llm"]))
    elif cfg.debug.mode == "preprocessor":
        processes.extend(launch_jobs(cfg, world_map, ["preprocessor", "preprocessor_llm"]))
    elif cfg.debug.mode == "actor+preprocessor":
        processes.extend(launch_jobs(cfg, world_map, ["actor", "environment", "actor_llm", "preprocessor", "preprocessor_llm"]))       
    elif cfg.debug.mode == "finetune+preprocessor":
        processes.extend(launch_jobs(cfg, world_map, ["finetune", "preprocessor", "preprocessor_llm"]))
    elif cfg.debug.mode in ["", "open_loop"]:
        processes.extend(launch_jobs(cfg, world_map))
    else:
        raise NotImplementedError(f"Unknown debug mode {cfg.debug.mode}")

    if os.environ.get("DRY_RUN", "0") == "1":
        assert not processes
        return
    watch_processes_running(exp_dir, processes, bool(cfg.debug.mode))


if __name__ == "__main__":
    main()
