"""
Test RC Actor - Standalone test for online reasoning/summarization rollouts

This script:
1. Loads configuration from a Hydra config file (default: conf/base.yaml)
2. Starts vLLM inference servers using the same logic as launch.py
3. Runs RC actor with online rollouts
4. Shows you where to find the output stream
"""
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _popen(
    cmd: list[str],
    env: dict | None = None,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    """Wrapper around subprocess.Popen (same as launch.py)"""
    return subprocess.Popen(
        cmd,
        env=env if env else os.environ,
        stdout=stdout,
        stderr=stderr,
        preexec_fn=os.setsid,  # Create new process group
    )


def save_command(log_dir: Path, cmd: list[str]):
    """Save command to file for debugging"""
    with open(log_dir / "command.txt", "w") as f:
        f.write(" ".join(cmd))


def start_actor_llm(
    cfg: DictConfig,
    actor_llm_idx: int,
    local_idx: int,
    gpus: list[int],
    exp_dir: Path
):
    """Start an actor LLM server (same logic as launch.py)"""
    
    finetune_model_path = exp_dir / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
    else:
        actor_model_path = cfg.model_path

    log_dir = exp_dir / f"actor_vllm_{actor_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    
    # For testing, we use simple OpenAI API server (no weight updates needed)
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(actor_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8080 + local_idx),
        "--seed",
        str(cfg.seed + actor_llm_idx),
    ]

    # Add vLLM kwargs as separate arguments
    if hasattr(cfg, 'vllm_config') and cfg.vllm_config.vllm_kwargs:
        for k, v in cfg.vllm_config.vllm_kwargs.items():
            cmd.append(f"--{k}")
            if v not in [None, ""]:
                cmd.append(str(v))

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Starting actor_llm {actor_llm_idx} with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        process = _popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
            stdout=log_file,
            stderr=err_file,
        )
    
    return process


def start_summarization_llm(
    cfg: DictConfig,
    summarization_llm_idx: int,
    local_idx: int,
    gpus: list[int],
    exp_dir: Path
):
    """Start a summarization LLM server (same logic as actor LLM but with different model/config)"""
    
    # Use summarization model if configured, otherwise use actor model
    if cfg.get('summarization_model_path'):
        summarization_model_path = cfg.summarization_model_path
    else:
        finetune_model_path = exp_dir / "finetune" / "current"
        if os.path.exists(finetune_model_path):
            summarization_model_path = finetune_model_path
        else:
            summarization_model_path = cfg.model_path

    log_dir = exp_dir / f"summarization_vllm_{summarization_llm_idx}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Use summarization-specific port (8280+)
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(summarization_model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(8280 + summarization_llm_idx),
        "--seed",
        str(cfg.seed + 1000 + summarization_llm_idx),  # Different seed space
    ]

    # Add vLLM kwargs - use summarization config if available, otherwise actor config
    vllm_config = cfg.get('summarization_vllm_config') if cfg.get('summarization_vllm_config') else cfg.vllm_config
    if hasattr(vllm_config, 'vllm_kwargs') and vllm_config.vllm_kwargs:
        for k, v in vllm_config.vllm_kwargs.items():
            cmd.append(f"--{k}")
            if v not in [None, ""]:
                cmd.append(str(v))

    gpu_str = ",".join([str(gpu) for gpu in gpus])
    logger.info(f"Starting summarization_llm {summarization_llm_idx} with command: {' '.join(cmd)} on gpus: {gpu_str}")
    save_command(log_dir, cmd)
    
    log_file_path = os.path.join(log_dir, "stdout.log")
    err_file_path = os.path.join(log_dir, "stderr.log")
    
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        process = _popen(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_str},
            stdout=log_file,
            stderr=err_file,
        )
    
    return process


def wait_for_llm_server(port: int, timeout: int = 300) -> bool:
    """Wait for LLM server to be ready"""
    import requests
    
    start_time = time.time()
    url = f"http://localhost:{port}/v1/models"
    
    logger.info(f"Waiting for LLM server on port {port} to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"LLM server on port {port} is ready!")
                return True
        except Exception:
            pass
        time.sleep(5)
    
    logger.error(f"LLM server on port {port} did not become ready in {timeout}s")
    return False


def start_environment(cfg: DictConfig, env_idx: int, port: int, exp_dir: Path, job_idx: int):
    """Start an environment server (same logic as launch.py)"""
    
    run_dir = exp_dir / f"environment_{env_idx}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config in the expected location for the environment to read
    config_dir = exp_dir / "conf"
    os.makedirs(config_dir, exist_ok=True)
    
    cmd = [
        "python",
        "-m",
        "pipelinerl.entrypoints.run_environment",
        "--config-dir",
        str(config_dir),
        "--config-name",
        "exp_config",
        f"output_dir={exp_dir}",
        f"hydra.run.dir={str(run_dir)}",
        f"me.job_idx={job_idx}",
    ]
    
    logger.info(f"Starting environment {env_idx} on port {port}")
    save_command(run_dir, cmd)
    
    log_file_path = run_dir / "stdout.log"
    err_file_path = run_dir / "stderr.log"
    
    with open(log_file_path, "a") as log_file, open(err_file_path, "a") as err_file:
        process = _popen(
            cmd,
            env=dict(os.environ),
            stdout=log_file,
            stderr=err_file,
        )
    
    return process


def wait_for_environment(port: int, timeout: int = 60) -> bool:
    """Wait for environment server to be ready"""
    import requests
    
    start_time = time.time()
    url = f"http://localhost:{port}/health"
    
    logger.info(f"Waiting for environment server on port {port} to be ready...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Environment server on port {port} is ready!")
                return True
        except Exception:
            pass
        time.sleep(3)
    
    logger.error(f"Environment server on port {port} did not become ready in {timeout}s")
    return False


def create_test_dataset() -> List[dict]:
    """Create a small test dataset for reasoning/summarization"""
    return [
        {
            "task": "What is 2 + 2?",
            "answer": "4",
            "dataset": "test_math",
            "id": 0,
        },
        {
            "task": "What is the capital of France?",
            "answer": "Paris",
            "dataset": "test_qa",
            "id": 1,
        },
        {
            "task": "Solve: x + 5 = 10. What is x?",
            "answer": "5",
            "dataset": "test_math",
            "id": 2,
        },
    ]




async def simple_rollout_policy(cfg, llm, problem, session):
    """Simple rollout policy for testing"""
    from pipelinerl.async_llm import llm_async_generate, make_training_text
    from pipelinerl.rollouts import RolloutResult, BaseMetrics
    from tapeagents.core import Prompt
    
    # Create a simple prompt
    messages = [
        {"role": "user", "content": problem["task"]}
    ]
    
    start_time = time.time()
    llm_call = await llm_async_generate(llm, Prompt(messages=messages), session)
    latency = time.time() - start_time
    
    # Create training text
    training_text = make_training_text(llm, llm_call)
    training_text.reward = 0.5  # Dummy reward
    
    # Create metrics
    metrics = BaseMetrics(
        success=False,
        no_error=True,
        no_answer=False,
    )
    
    return RolloutResult(
        training_texts=[training_text],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset", "test"),
    )




def get_actor_urls(num_llms: int, gpus_per_llm: int = 1) -> list[str]:
    """Get actor LLM URLs (same pattern as world_map.get_actor_urls() in launch.py)
    
    Port is determined by the minimum GPU ID allocated to each LLM (local_idx).
    For gpus_per_llm=1: ports are 8080, 8081, 8082, ...
    For gpus_per_llm=2: ports are 8080, 8082, 8084, ... (using GPU 0, 2, 4, ...)
    """
    urls = []
    for i in range(num_llms):
        local_idx = i * gpus_per_llm  # Minimum GPU ID for this LLM
        urls.append(f"http://localhost:{8080 + local_idx}")
    return urls


def prepare_config_for_test(cfg: DictConfig, output_dir: Path, num_llms: int, gpus_per_llm: int = 1, num_summarization_llms: int = 0, summarization_gpus_per_llm: int = 1, num_envs: int = 1, env_start_port: int = 9000):
    """Prepare config for testing. Mainly for setting the llm_urls and job info. Rest everything is set in the config file."""
    
    # Set output directory
    cfg.output_dir = str(output_dir)
    
    # Create LLM URLs (same pattern as launch.py: line 187, 198)
    llm_urls = "+".join(get_actor_urls(num_llms, gpus_per_llm))
    
    # Use OmegaConf to properly set llm_urls (disable struct mode temporarily)
    if not OmegaConf.select(cfg, 'me'):
        OmegaConf.update(cfg, 'me', {})
    
    # Temporarily disable struct mode to add llm_urls
    OmegaConf.set_struct(cfg.me, False)
    cfg.me.llm_urls = llm_urls
    
    # Create separate summarization LLM URLs if configured
    if num_summarization_llms > 0:
        # Summarization LLMs start at a different base port (e.g., 8280)
        summarization_urls = []
        base_gpu_offset = num_llms * gpus_per_llm  # Start after actor GPUs
        for i in range(num_summarization_llms):
            local_idx = base_gpu_offset + (i * summarization_gpus_per_llm)
            summarization_urls.append(f"http://localhost:{8280 + i}")
        cfg.me.summarization_llm_urls = "+".join(summarization_urls)
    
    OmegaConf.set_struct(cfg.me, True)
    
    # Create job information for environments (needed by rollout functions)
    jobs = []
    job_idx = 0
    
    # Add environment jobs
    for env_idx in range(num_envs):
        jobs.append({
            "kind": "environment",
            "idx": job_idx,
            "replica_idx": env_idx,
            "local_idx": 0,
            "node_rank": 0,
            "hostname": "localhost",
            "port": env_start_port + env_idx,
            "gpus": [],
            "url": "",
        })
        job_idx += 1
    
    # Add to config
    OmegaConf.set_struct(cfg, False)
    cfg.jobs = jobs
    OmegaConf.set_struct(cfg, True)
    
    return cfg


@hydra.main(config_path="../conf", config_name="base", version_base="1.3.2")
def main(cfg: DictConfig):
    """Main test function"""
    
    # Calculate number of LLMs and GPU allocation from config (similar to WorldMap)
    import torch
    
    # Get vLLM parallelism config
    llm_kwargs = cfg.vllm_config.vllm_kwargs
    tp = llm_kwargs.get("tensor-parallel-size", 1)
    pp = llm_kwargs.get("pipeline-parallel-size", 1)
    gpus_per_llm = tp * pp
    
    # For testing, we use actor fraction from config
    num_llms = cfg.get('test_world', {}).get('actor_fraction', 1)
    num_summarization_llms = cfg.get('test_world', {}).get('summarization_fraction', 0)
    
    # Get summarization vLLM config (if different from actor)
    if cfg.get('summarization_vllm_config') and cfg.summarization_vllm_config.get('vllm_kwargs'):
        summarization_llm_kwargs = cfg.summarization_vllm_config.vllm_kwargs
        summarization_tp = summarization_llm_kwargs.get("tensor-parallel-size", 1)
        summarization_pp = summarization_llm_kwargs.get("pipeline-parallel-size", 1)
        summarization_gpus_per_llm = summarization_tp * summarization_pp
    else:
        summarization_gpus_per_llm = gpus_per_llm
    
    # Allocate GPUs: each LLM gets gpus_per_llm consecutive GPUs
    gpu_ids = []
    for llm_idx in range(num_llms):
        llm_gpus = list(range(llm_idx * gpus_per_llm, (llm_idx + 1) * gpus_per_llm))
        gpu_ids.extend(llm_gpus)
    
    # Allocate GPUs for summarization LLMs (after actor LLMs)
    summarization_gpu_ids = []
    if num_summarization_llms > 0:
        base_gpu_offset = num_llms * gpus_per_llm
        for llm_idx in range(num_summarization_llms):
            llm_gpus = list(range(base_gpu_offset + llm_idx * summarization_gpus_per_llm, 
                                base_gpu_offset + (llm_idx + 1) * summarization_gpus_per_llm))
            summarization_gpu_ids.extend(llm_gpus)
    
    # Create output directory
    output_dir = Path(cfg.output_dir + "_" + str(int(time.time())))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Environment configuration
    num_envs = cfg.test_world.env_replicas
    env_start_port = cfg.test_world.environment_start_port
    actor_group_port = cfg.test_world.actor_group_port
    
    logger.info("=" * 80)
    logger.info("RC Actor Test Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_path}")
    logger.info(f"GPUs per LLM: {gpus_per_llm}")
    logger.info(f"Number of LLM servers: {num_llms}")
    logger.info(f"Number of summarization LLM servers: {num_summarization_llms}")
    logger.info(f"Number of environment servers: {num_envs}")
    logger.info(f"Number of reasoning steps: {cfg.actor.num_reasoning_steps}")
    logger.info(f"Actor GPU allocation: {gpu_ids}")
    if num_summarization_llms > 0:
        logger.info(f"Summarization GPU allocation: {summarization_gpu_ids}")
        logger.info(f"Summarization model: {cfg.get('summarization_model_path', cfg.model_path)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Prepare config (includes job information)
    cfg = prepare_config_for_test(cfg, output_dir, num_llms, gpus_per_llm, num_summarization_llms, summarization_gpus_per_llm, num_envs, env_start_port)
    
    # Save config (environments need to read this)
    config_dir = output_dir / "conf"
    os.makedirs(config_dir, exist_ok=True)
    config_file = config_dir / "exp_config.yaml"
    with open(config_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved config to {config_file}")
    
    # Start servers and track their ports
    processes = []
    llm_ports = []
    env_ports = []
    
    try:
        # Step 1: Start environment servers
        logger.info("=" * 80)
        logger.info("Step 1: Starting environment servers")
        logger.info("=" * 80)
        
        for env_idx in range(num_envs):
            port = env_start_port + env_idx
            env_ports.append(port)
            # job_idx for environments (they come first in the jobs list)
            job_idx = env_idx
            logger.info(f"Starting environment {env_idx} on port {port}")
            process = start_environment(cfg, env_idx, port, output_dir, job_idx)
            processes.append(process)
        
        # Wait for environment servers to be ready
        logger.info("\nWaiting for environment servers to be ready...")
        all_env_ready = True
        for port in env_ports:
            if not wait_for_environment(port, timeout=60):
                all_env_ready = False
                logger.error(f"Environment server on port {port} failed to start!")
        
        if not all_env_ready:
            logger.error("Not all environment servers started successfully!")
            return 1
        
        logger.info("✅ All environment servers are ready!")
        
        # Step 2: Start vLLM servers
        logger.info("=" * 80)
        logger.info("Step 2: Starting vLLM servers")
        logger.info("=" * 80)
        
        for i in range(num_llms):
            # Allocate GPUs for this LLM (gpus_per_llm consecutive GPUs)
            llm_gpus = list(range(i * gpus_per_llm, (i + 1) * gpus_per_llm))
            # Use minimum GPU ID as the local_idx (for port assignment)
            local_idx = min(llm_gpus) if llm_gpus else i
            port = 8080 + local_idx
            llm_ports.append(port)
            logger.info(f"Starting LLM {i} on GPUs {llm_gpus}, port {port}")
            process = start_actor_llm(cfg, i, local_idx, llm_gpus, output_dir)
            processes.append(process)
        
        # Wait for all LLM servers to be ready
        logger.info("\nWaiting for all LLM servers to be ready...")
        all_llm_ready = True
        for port in llm_ports:
            if not wait_for_llm_server(port, timeout=300):
                all_llm_ready = False
                logger.error(f"LLM server on port {port} failed to start!")
        
        if not all_llm_ready:
            logger.error("Not all LLM servers started successfully!")
            return 1
        
        logger.info("✅ All vLLM servers are ready!")
        
        # Step 3: Start summarization LLM servers (if configured)
        summarization_llm_ports = []
        if num_summarization_llms > 0:
            logger.info("=" * 80)
            logger.info("Step 3: Starting summarization LLM servers")
            logger.info("=" * 80)
            
            base_gpu_offset = num_llms * gpus_per_llm
            for i in range(num_summarization_llms):
                # Allocate GPUs for this summarization LLM
                llm_gpus = list(range(base_gpu_offset + i * summarization_gpus_per_llm,
                                    base_gpu_offset + (i + 1) * summarization_gpus_per_llm))
                port = 8280 + i
                summarization_llm_ports.append(port)
                logger.info(f"Starting summarization LLM {i} on GPUs {llm_gpus}, port {port}")
                process = start_summarization_llm(cfg, i, i, llm_gpus, output_dir)
                processes.append(process)
            
            # Wait for all summarization LLM servers to be ready
            logger.info("\nWaiting for all summarization LLM servers to be ready...")
            all_summarization_llm_ready = True
            for port in summarization_llm_ports:
                if not wait_for_llm_server(port, timeout=300):
                    all_summarization_llm_ready = False
                    logger.error(f"Summarization LLM server on port {port} failed to start!")
            
            if not all_summarization_llm_ready:
                logger.error("Not all summarization LLM servers started successfully!")
                return 1
            
            logger.info("✅ All summarization LLM servers are ready!")
        
        # Run RC actor
        logger.info("=" * 80)
        step_num = 4 if num_summarization_llms > 0 else 3
        logger.info(f"Step {step_num}: Running RC Actor")
        logger.info("=" * 80)
        
        # Import and run
        from pipelinerl import rc_actor
        
        # Run actor loop (it will set streams backend internally)
        logger.info("Starting RC actor loop...")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Stream file: {output_dir}/streams/actor/0/0/0.jsonl")
        
        rc_actor.run_actor_loop(cfg)
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"\nTest failed with exception: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup: Stop all servers
        logger.info("=" * 80)
        logger.info("Cleanup: Stopping all servers")
        logger.info("=" * 80)
        
        for process in processes:
            if process:
                try:
                    logger.info(f"Stopping process {process.pid}")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=10)
                except Exception as e:
                    logger.warning(f"Error stopping process: {e}")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except:
                        pass
        
        logger.info(f"\nTest artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
