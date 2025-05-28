import time
import random

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.world import Job
from tapeagents.core import Prompt, TrainingText
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate
from pipelinerl.finetune.data import MASKED_TOKEN_ID
from pipelinerl.math.verifier_api import verify_answer_rpc


class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float


class RolloutResult(BaseModel):
    training_texts: list[TrainingText]
    metrics: dict[str, float]
    latency: float
    # optional so fields that it can be filled later after RolloutResult is created
    model_version: int | None = None
    dataset_name: str | None = None
    group_id: str | None = None


def make_prompt(problem: dict, cfg: DictConfig) -> Prompt:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    return Prompt(messages=messages)


async def generate_math_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    prompt = make_prompt(problem=problem, cfg=cfg)
    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    rewards = RewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor

    # math_verify is a fast environment, no support for environment replicas for now
    env_jobs = [Job(**job) for job in cfg.jobs if job["kind"] == "environment"]
    # choose the job randomly
    env_job = random.choice(env_jobs)
    assert env_job.port is not None
    answer_status = await verify_answer_rpc(
        session=session,
        host=env_job.hostname,
        port=env_job.port,
        prediction=llm_call.output.content,
        gold=problem["answer"],
        strict=True,
    )

    trace = llm.make_training_text(llm_call.prompt, llm_call.output)

    input_ids = [lp.token_id for lp in llm_call.logprobs]
    labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]

    # Check if the generation is finished (ended with EOS token)
    finished = 1 if input_ids[-1] == llm.tokenizer.eos_token_id else 0

    # Determine reward based on answer status and finished state
    match (answer_status, finished):
        case ("wrong", 0):
            reward = rewards.wrong_answer_not_finished
        case ("wrong", 1):
            reward = rewards.wrong_answer_finished
        case ("no_answer", 0):
            reward = rewards.no_answer_not_finished
        case ("no_answer", 1):
            reward = rewards.no_answer_finished
        case ("unparsable", 0):
            reward = rewards.unparsable_not_finished
        case ("unparsable", 1):
            reward = rewards.unparsable_finished
        case ("correct", 0):
            reward = rewards.correct_answer_not_finished
        case ("correct", 1):
            reward = rewards.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{finished}")

    # Apply discount factor based on output length
    reward *= discount_factor**llm_call.output_length_tokens

    # Apply masking to input tokens that aren't generated
    labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels

    trace.input_ids = input_ids
    trace.labels = labels
    trace.reward = reward
    trace.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]

    metrics = {
        "reward": reward,
        "success": answer_status == "correct",
        "no_error": answer_status != "unparsable",
        "no_answer": answer_status == "no_answer",
        "prompt_tokens": llm_call.prompt_length_tokens,
        "output_tokens": llm_call.output_length_tokens,
        "overflow": 0 if finished else 1,
    }

    return RolloutResult(training_texts=[trace], metrics=metrics, latency=latency, dataset_name=problem.get("dataset"))
