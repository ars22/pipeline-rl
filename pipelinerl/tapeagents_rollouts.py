import asyncio
import time

import aiohttp
from omegaconf import DictConfig
from tapeagents.agent import Agent, LLMEvent, LLMStream
from tapeagents.core import Prompt, StopStep
from tapeagents.dialog_tape import DialogTape, UserStep
from tapeagents.environment import Environment
from tapeagents.llms.trainable import TrainableLLM
from tapeagents.orchestrator import get_agent_and_env_from_config, main_loop

from pipelinerl.rollouts import RolloutResult, TrainingText


def run_tapeagent(
    task: str, agent: Agent, environment: Environment, max_loops: int
) -> tuple[list[TrainingText], dict[str, float]]:
    start_tape = DialogTape(steps=[UserStep(content=task)])
    tape: DialogTape | None = None
    for event in main_loop(agent, start_tape, environment, max_loops):
        if event.agent_tape:
            tape = event.agent_tape
        elif event.env_tape:
            tape = event.env_tape
    assert tape is not None, "No tape generated"
    has_errors = any([1 for s in tape.steps if s.llm_dict().get("error")])
    has_answer = any([isinstance(s, StopStep) for s in tape.steps])
    _, llm_calls = agent.reuse(tape)
    samples = [agent.make_training_text(llm_call) for llm_call in llm_calls]
    reward = 0  # TODO: implement verifier usage and reward calculation
    metrics = {
        "reward": reward,
        "success": reward > 0,
        "no_error": not has_errors,
        "no_answer": not has_answer,
        "prompt_tokens": sum([llm_call.prompt_length_tokens for llm_call in llm_calls]),
        "output_tokens": sum([llm_call.output_length_tokens for llm_call in llm_calls]),
        "overflow": 0,  # TODO: should we treat max_loops stop as overflow?
    }
    return samples, metrics


async def generate_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    time_start = time.time()
    task: str = cfg.actor.task_template.format(task=problem["task"])
    agent, environment = get_agent_and_env_from_config(cfg)
    
    # set the LLM directly on the agent
    agent.llms = {"default": llm}
    
    samples, metrics = await asyncio.to_thread(run_tapeagent, task, agent, environment, cfg.max_loops)
    latency = time.time() - time_start
    return RolloutResult(training_texts=samples, metrics=metrics, latency=latency, dataset_name=problem.get("dataset"))
