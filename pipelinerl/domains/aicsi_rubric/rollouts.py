"""
Rollout function for the AI-CoScientist rubric (aicsi_rubric) domain.

Generates a response to a scientific query and scores it against a binary
rubric using an LLM judge. Reward = n_yes / n_criteria (normalized to [0, 1])
with an optional length discount applied.
"""

import time
import os

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .verifier_api import verify_rubric

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def remove_reasoning(completion: str, reasoning_delimiters: list[str] | None = None) -> str:
    """Strip reasoning prefix (e.g. </think> content) from a completion."""
    if not reasoning_delimiters:
        return completion
    for delim in reasoning_delimiters:
        if delim in completion:
            return completion.split(delim)[-1].strip()
    return ""


class Metrics(BaseMetrics):
    penalty: float


async def generate_aicsi_rubric_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """
    Generate a response to the scientific query and score it against the rubric.

    problem dict keys (produced by load_datasets.py):
      - "task":    the scientific query
      - "rubric":  list of criterion dicts (parsed rubric_criteria)
      - "title":   paper title (optional, for logging)
      - "dataset": source dataset name
    """
    messages = []
    if cfg.actor.system_prompt is not None:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": cfg.actor.task_template.format(task=problem["task"])})
    prompt = Prompt(messages=messages)

    time_start = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - time_start

    assert llm_call.output.content is not None
    generation_raw = llm_call.output.content

    # Strip reasoning tokens from thinking models
    reasoning_delimiters = (
        cfg.llm_grader.reasoning_delimiters
        if "reasoning_delimiters" in cfg.llm_grader
        else None
    )
    generation_final = remove_reasoning(generation_raw, reasoning_delimiters=reasoning_delimiters)
    if not generation_final.strip():
        generation_final = generation_raw

    trace = make_training_text(llm, llm_call)
    discount_factor = cfg.actor.discount_factor

    llm_grader_cfg = cfg.get("llm_grader", None)
    wandb_table_cfg = llm_grader_cfg.get("wandb_table", None) if llm_grader_cfg is not None else None
    wandb_table_enabled = bool(wandb_table_cfg.get("enabled", True)) if wandb_table_cfg is not None else True

    verification = await verify_rubric(
        query=problem["task"],
        rubric=problem["rubric"],
        generation=generation_final,
        prompt_name=getattr(cfg.llm_grader, "prompt_name", None),
        model=(
            getattr(cfg.llm_grader, "name", None)
            if (
                "/" in getattr(cfg.llm_grader, "name", "")
                or getattr(cfg.llm_grader, "backend", "openai") != "openai"
            )
            else os.getenv("HF_ENDPOINT_REPO")
        ),
        sampling_kwargs=getattr(cfg.llm_grader, "sampling_kwargs", None),
        log_wandb_metrics=cfg.wandb.use_wandb,
        collect_table_entry=bool(cfg.wandb.use_wandb and wandb_table_enabled),
        backend=getattr(cfg.llm_grader, "backend", "openai"),
    )

    # Reward = rubric score (n_yes / n_criteria) with length discount
    reward = verification.score * (discount_factor ** llm_call.output_length_tokens)
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=verification.score >= 0.7,  # ≥70% rubric criteria satisfied = success
        no_error=True,
        no_answer=False,
        penalty=0.0,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        verifier_metrics=verification.metrics,
        verifier_table_entry=verification.table_entry,
    )
