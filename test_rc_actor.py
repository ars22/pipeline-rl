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


def load_test_dataset(dataset_names, **kwargs):
    """Dataset loader for testing"""
    return create_test_dataset()


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




def prepare_config_for_test(cfg: DictConfig, output_dir: Path, num_llms: int):
    """Prepare config for testing"""
    
    # Set output directory
    cfg.output_dir = str(output_dir)
    
    # Create LLM URLs
    llm_urls = "+".join([f"http://localhost:{8080 + i}/v1" for i in range(num_llms)])
    if not hasattr(cfg, 'me'):
        cfg.me = {}
    cfg.me.llm_urls = llm_urls
    
    # Set test dataset loader
    cfg.dataset_loader = "test_rc_actor.load_test_dataset"
    cfg.train_dataset_names = ["test"]
    cfg.test_dataset_names = []
    
    # Enable debug mode
    if not hasattr(cfg, 'debug'):
        cfg.debug = {}
    cfg.debug.mode = True
    
    # Set RC actor config
    if not hasattr(cfg, 'actor'):
        cfg.actor = {}
    
    # Set rollout policies to test policy
    cfg.actor.solution_rollout_policy = "test_rc_actor.simple_rollout_policy"
    cfg.actor.summarization_rollout_policy = "test_rc_actor.simple_rollout_policy"
    
    # Ensure we have reasoning/summarization templates
    if not hasattr(cfg.actor, 'reasoning_prompt_template'):
        cfg.actor.reasoning_prompt_template = (
            "Solve this problem step by step.\n\n"
            "Problem: {problem}\n\n"
            "Previous work: {curr_summary}\n\n"
            "Continue your reasoning:"
        )
    
    if not hasattr(cfg.actor, 'summarization_prompt_template'):
        cfg.actor.summarization_prompt_template = (
            "Summarize the solution progress.\n\n"
            "Problem: {problem}\n\n"
            "Previous summary: {existing_summary}\n\n"
            "New reasoning: {reasoning}\n\n"
            "Provide a concise updated summary:"
        )
    
    # Disable wandb
    if not hasattr(cfg, 'wandb'):
        cfg.wandb = {}
    cfg.wandb.use_wandb = False
    
    # Set streams backend
    if not hasattr(cfg, 'streams'):
        cfg.streams = {}
    cfg.streams.backend = "files"
    
    # Disable eval
    cfg.eval_every_n_versions = 0
    
    # Set finetune config for state management
    if not hasattr(cfg, 'finetune'):
        cfg.finetune = {}
    cfg.finetune.max_lag = None
    
    return cfg


@hydra.main(config_path="../conf", config_name="base", version_base="1.3.2")
def main(cfg: DictConfig):
    """Main test function"""
    
    # Get test parameters from environment or use defaults
    num_llms = int(os.getenv("TEST_NUM_LLMS", "2"))
    num_reasoning_steps = int(os.getenv("TEST_NUM_STEPS", "2"))
    gpu_ids_str = os.getenv("TEST_GPUS", "0,1")
    gpu_ids = [int(x) for x in gpu_ids_str.split(",")]
    
    # Create output directory
    output_dir = Path("/tmp/test_rc_actor_" + str(int(time.time())))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("RC Actor Test Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {cfg.model_path}")
    logger.info(f"Number of LLM servers: {num_llms}")
    logger.info(f"Number of reasoning steps: {num_reasoning_steps}")
    logger.info(f"GPU IDs: {gpu_ids}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    # Prepare config
    cfg = prepare_config_for_test(cfg, output_dir, num_llms)
    
    # Override num_reasoning_steps if specified
    cfg.actor.num_reasoning_steps = num_reasoning_steps
    
    # Save config
    config_file = output_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saved config to {config_file}")
    
    # Start vLLM servers
    processes = []
    
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Step 1: Starting vLLM servers")
        logger.info("=" * 80)
        
        for i in range(num_llms):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            process = start_actor_llm(cfg, i, i, [gpu_id], output_dir)
            processes.append(process)
        
        # Wait for all servers to be ready
        logger.info("\nWaiting for all servers to be ready...")
        all_ready = True
        for i in range(num_llms):
            port = 8080 + i
            if not wait_for_llm_server(port, timeout=300):
                all_ready = False
                logger.error(f"Server on port {port} failed to start!")
        
        if not all_ready:
            logger.error("Not all servers started successfully!")
            return 1
        
        logger.info("\n✅ All vLLM servers are ready!")
        
        # Run RC actor
        logger.info("\n" + "=" * 80)
        logger.info("Step 2: Running RC Actor")
        logger.info("=" * 80)
        
        # Import and run
        from pipelinerl import rc_actor
        from pipelinerl.streams import set_streams_backend
        
        # Set streams backend
        set_streams_backend(**cfg.streams)
        
        # Run actor loop
        logger.info("Starting RC actor loop...")
        rc_actor.run_actor_loop(cfg)
        
        # Done!
        logger.info("\n" + "=" * 80)
        logger.info("✅ RC Actor Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"\nOutput directory: {output_dir}")
        logger.info(f"Stream file: {output_dir}/streams/actor/0/0/0.jsonl")
        logger.info(f"\nTo view rollouts:")
        logger.info(f"  tail -f {output_dir}/streams/actor/0/0/0.jsonl")
        logger.info(f"  # or")
        logger.info(f"  cat {output_dir}/streams/actor/0/0/0.jsonl | jq .")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"\nTest failed with exception: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup: Stop all vLLM servers
        logger.info("\n" + "=" * 80)
        logger.info("Cleanup: Stopping vLLM servers")
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
