import time
import random
from typing import Optional

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.world import Job
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import verify_answer_rpc_genrm, verify_answer_rpc

class Metrics(BaseMetrics):
    penalty: float = 0.0
    outcome_reward: Optional[float] = None
    genrm_original_score: Optional[float] = None
    genrm_normalized_score: Optional[float] = None

class RewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float
    buffer_tokens: int = 0 # 0 means no overlong reward shaping

def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.

async def generate_math_rollout_genrm(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> Optional[RolloutResult]:
    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

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
    use_genrm = cfg.world.get("genrm_fraction", 0) > 0 and 'solution' in problem
    prompt_template_path = cfg.environment.get("prompt_template_path", "")
    
    print(f"Using GenRM: {use_genrm} for problem: {problem['dataset']}")
    if use_genrm:
        answer_status, genrm_score = await verify_answer_rpc_genrm(
            cfg=cfg,
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
            prompt=problem["task"],
            solution=problem["solution"],
            prompt_template_path=prompt_template_path,
            return_score=True,   # @TODO: hardcoded to be True for GenRM, make it more robust later
        )
    else:    # for validation dataset, no need to call GenRM
        answer_status = await verify_answer_rpc(
            session=session,
            host=env_job.hostname,
            port=env_job.port,
            prediction=llm_call.output.content,
            gold=problem["answer"],
            strict=True,
        )
    trace = make_training_text(llm, llm_call)
    # Determine reward based on answer status and finished state
    match (answer_status, trace.finished):
        case ("wrong", False):
            outcome_reward = rewards.wrong_answer_not_finished
        case ("wrong", True):
            outcome_reward = rewards.wrong_answer_finished
        case ("no_answer", False):
            outcome_reward = rewards.no_answer_not_finished
        case ("no_answer", True):
            outcome_reward = rewards.no_answer_finished
        case ("unparsable", False):
            outcome_reward = rewards.unparsable_not_finished
        case ("unparsable", True):
            outcome_reward = rewards.unparsable_finished
        case ("correct", False):
            outcome_reward = rewards.correct_answer_not_finished
        case ("correct", True):
            outcome_reward = rewards.correct_answer_finished
        case _:
            raise ValueError(f"Invalid answer_status/finished combination: {answer_status}/{trace.finished}")

    if use_genrm:    # this is only happening for training dataset
        if genrm_score:  
            if cfg.actor.genrm_norm_method == "smooth":
                genrm_normalized_score = (genrm_score - 1) / 4
            else:
                raise ValueError(f"Invalid genrm norm method: {cfg.actor.genrm_norm_method}")
            reward = genrm_normalized_score
            # Apply discount factor based on output length
            reward *= discount_factor**llm_call.output_length_tokens
            overlong_penalty = 0
            if rewards.buffer_tokens > 0:
                overlong_penalty = length_penalty(llm.parameters['max_tokens'], llm_call.output_length_tokens, rewards.buffer_tokens)
            reward += overlong_penalty
            trace.reward = reward

            metrics = Metrics(
                reward=reward,
                outcome_reward=outcome_reward,
                genrm_original_score=genrm_score,
                genrm_normalized_score=genrm_normalized_score,
                success=answer_status == "correct",
                no_error=answer_status != "unparsable",
                no_answer=answer_status == "no_answer",
                penalty=overlong_penalty,
            )

            return RolloutResult(
                training_texts=[trace],
                metrics=metrics,
                latency=latency, 
                dataset_name=problem.get("dataset"),
            )
        else:
            return None    # GenRM score is not available, return None to signal the rollout is not valid
    else:   # for validation dataset, but we still need to collect the rollouts
        reward = outcome_reward
        trace.reward = reward
        metrics = Metrics(
            reward=reward,
            success=answer_status == "correct",
            no_error=answer_status != "unparsable",
            no_answer=answer_status == "no_answer",
        )
        return RolloutResult(
            training_texts=[trace],
            metrics=metrics,
            latency=latency, 
            dataset_name=problem.get("dataset"),
        )