
import logging
import os
import random
import time
import aiohttp
from hydra.utils import instantiate
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult
from pipelinerl.world import Job
from tapeagents.agent import Agent
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.remote_environment import AsyncRemoteEnvironment
from tapeagents.orchestrator import async_execute_agent
from tapeagents.io import save_json_tape
from pipelinerl.miniwob.agent import WebTape


logger = logging.getLogger(__name__)


async def generate_miniwob_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    ### STARTING POINT #1

    # choose a random environment server
    # Generate environment
    # Generate TapeAgent
    # run the agent
    # get llm calls from tape
    # compute rewards
    # get training text from llm calls

    start_time = time.time()

    # (1) Choose a random environment server
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the env job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    env_job_url = f"http://{env_job.hostname}:{env_job.port}"

    # (2) Generate environment, TapeAgent, and run them to get a Tape
    environment = AsyncRemoteEnvironment(server_url=env_job_url)  # type: ignore
    async with environment.acontext(session, wait_for_env=True) as env:
        tape_dict, _ = await env.start_task(problem)
        tape: WebTape = WebTape(**tape_dict)  # convert http response dict to WebTape object
        t = time.perf_counter()
        try:
            actions = await env.a_actions()
            tools_description = await env.a_tools_description()
            logger.info(f"Available tools: {tools_description}")
            agent: Agent = instantiate(cfg.agent, known_actions=actions, tools_description=tools_description)
            tape = await async_execute_agent(agent, tape, env, session, max_loops=cfg.agent_max_loops)
        except Exception as e:
            logger.error(f"Error occurred while running agent: {e}")
        tape.metadata.result = {"execution_time": time.perf_counter() - t}

    # save the tape as we go
    save_json_tape(tape, os.path.join(cfg.output_dir, "tapes"), tape.metadata.id)

    # TODO: continue...

    return RolloutResult(
        training_texts=[],
        metrics={},
        latency=time.time() - start_time,
        dataset_name=problem["dataset"],
    )

