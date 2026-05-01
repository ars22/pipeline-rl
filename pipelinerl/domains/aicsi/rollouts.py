"""
Rollout function for the AI-CoScientist (aicsi) domain.

Generates a response to a scientific query and scores it against the oracle
using the LLM-based similarity verifier (verify_aicsi). The reward is the
SIMILARITY_SCORE (0.0–1.0) with an optional length discount applied.
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
from .verifier_api import verify_aicsi

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


async def generate_aicsi_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """
    Generate a response to the scientific query and evaluate it against the oracle.

    problem dict keys (produced by load_datasets.py):
      - "task":    the scientific query
      - "answer":  the oracle (ground-truth) response
      - "title":   optional short title for the grader prompt (can be empty)
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

    # Strip reasoning (e.g. <think>...</think>) from thinking models
    reasoning_delimiters = (
        cfg.llm_grader.reasoning_delimiters
        if "reasoning_delimiters" in cfg.llm_grader
        else None
    )
    generation_final = remove_reasoning(generation_raw, reasoning_delimiters=reasoning_delimiters)
    # Fall back to raw if stripping produced empty output (non-thinking model)
    if not generation_final.strip():
        generation_final = generation_raw

    trace = make_training_text(llm, llm_call)
    discount_factor = cfg.actor.discount_factor

    llm_grader_cfg = cfg.get("llm_grader", None)
    wandb_table_cfg = llm_grader_cfg.get("wandb_table", None) if llm_grader_cfg is not None else None
    wandb_table_enabled = bool(wandb_table_cfg.get("enabled", True)) if wandb_table_cfg is not None else True

    # Use title from problem dict for PAPER_TITLE field in grader prompt
    query_for_grader = problem.get("title") or problem["task"]

    verification = await verify_aicsi(
        query=query_for_grader,
        oracle=problem["answer"],
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

    # Reward = similarity score (0.0–1.0) with length discount
    reward = verification.score * (discount_factor ** llm_call.output_length_tokens)
    trace.reward = reward

    metrics = Metrics(
        reward=reward,
        success=verification.score >= 0.9,  # treat ≥0.9 as success (equivalent to score 7/7 in proof)
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
