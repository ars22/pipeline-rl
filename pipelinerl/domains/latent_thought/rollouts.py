"""
Rollout function for the latent_thought domain.

Given (prefix, suffix) sliced from a raw text row, the policy generates an
auxiliary string `aux`. A frozen evaluator vLLM scores both the baseline
continuation `suffix | prefix` and the treatment continuation
`(aux ⊕ suffix) | prefix`. Reward is the difference of average per-token NLL.

Reward:
  avg_NLL_baseline  = -sum_logprob(suffix | prefix)        / |suffix|
  avg_NLL_treatment = -sum_logprob(aux⊕suffix | prefix)    / (|aux| + |suffix|)
  reward            = avg_NLL_baseline - avg_NLL_treatment

A positive reward means the auxiliary text lowered the average per-token NLL
of the overall continuation under the frozen evaluator.

Known v0 limitation: the reward admits an aux=suffix copy degeneracy
(see /docs/plans/latent-thought-domain.md). suffix_overlap_ratio is logged
per rollout for monitoring; it is NOT subtracted from the reward in v00.00.
"""

import asyncio
import difflib
import logging
import random
import time

import aiohttp
from omegaconf import DictConfig
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.rollouts import RolloutResult, BaseMetrics

logger = logging.getLogger(__name__)


class Metrics(BaseMetrics):
    avg_nll_baseline: float = 0.0
    avg_nll_treatment: float = 0.0
    nll_delta: float = 0.0
    aux_tokens: int = 0
    prefix_tokens: int = 0
    suffix_tokens: int = 0
    cut_offset: int = 0
    suffix_overlap_ratio: float = 0.0
    skipped: bool = False


def remove_reasoning(completion: str, reasoning_delimiters: list[str] | None = None) -> str:
    """Strip reasoning prefix (e.g. </think> content) from a completion."""
    if not reasoning_delimiters:
        return completion
    for delim in reasoning_delimiters:
        if delim in completion:
            return completion.split(delim)[-1].strip()
    return ""


def _sum_logprobs(result_dict: dict) -> float:
    """Extract sum of logprobs from a get_batch_logprobs_token_ids result entry.

    Response shape: {"content": [{"logprob": float, "token_id": str, ...}, ...]}
    """
    return sum(item["logprob"] for item in result_dict["content"])


def _truncate_to_fit(
    prefix_ids: list[int],
    suffix_ids: list[int],
    max_total: int,
) -> tuple[list[int], list[int]]:
    """Trim the longer of prefix/suffix until prefix+suffix fits within max_total.

    Truncates prefix from the LEFT (drop oldest tokens) and suffix from the RIGHT
    (drop trailing tokens). Returns (prefix_ids, suffix_ids).
    """
    while len(prefix_ids) + len(suffix_ids) > max_total:
        if len(prefix_ids) >= len(suffix_ids) and len(prefix_ids) > 0:
            prefix_ids = prefix_ids[1:]
        elif len(suffix_ids) > 0:
            suffix_ids = suffix_ids[:-1]
        else:
            break
    return prefix_ids, suffix_ids


async def generate_latent_thought_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
    *,
    evaluator_llm: TrainableLLM,
) -> RolloutResult:
    """Latent-thought rollout. See module docstring for the reward formula."""
    time_start = time.time()

    lt_cfg = cfg.get("latent_thought", {}) or {}
    max_total_tokens: int = int(lt_cfg.get("max_total_tokens", 16384))
    min_side_tokens: int = int(lt_cfg.get("min_side_tokens", 64))

    # ----- 1. Pick a cut online ---------------------------------------------
    cut_offsets = problem.get("cut_offsets") or []
    text = problem["text"]
    if not cut_offsets:
        # Loader should have filtered these out, but be defensive.
        return _make_skip_result(latency=time.time() - time_start, dataset=problem.get("dataset"))
    cut = random.choice(list(cut_offsets))
    prefix_text = text[:cut]
    suffix_text = text[cut:]

    # ----- 2. Token-level enforcement ---------------------------------------
    evaluator_llm.load_tokenizer()
    eval_tokenizer = evaluator_llm.tokenizer

    prefix_ids = eval_tokenizer(prefix_text, add_special_tokens=True).input_ids
    suffix_ids = eval_tokenizer(suffix_text, add_special_tokens=False).input_ids

    if len(prefix_ids) + len(suffix_ids) > max_total_tokens:
        prefix_ids, suffix_ids = _truncate_to_fit(prefix_ids, suffix_ids, max_total_tokens)

    if len(prefix_ids) < min_side_tokens or len(suffix_ids) < min_side_tokens:
        return _make_skip_result(
            latency=time.time() - time_start,
            dataset=problem.get("dataset"),
            cut_offset=cut,
            prefix_tokens=len(prefix_ids),
            suffix_tokens=len(suffix_ids),
        )

    # Re-derive text from token ids in case we truncated; the policy sees the
    # same text the evaluator will score over.
    prefix_text_view = eval_tokenizer.decode(prefix_ids, skip_special_tokens=False)
    suffix_text_view = eval_tokenizer.decode(suffix_ids, skip_special_tokens=False)

    # ----- 3. Build policy prompt and generate aux --------------------------
    messages = []
    if cfg.actor.system_prompt is not None:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({
        "role": "user",
        "content": cfg.actor.task_template.format(
            prefix=prefix_text_view,
            suffix=suffix_text_view,
        ),
    })
    prompt = Prompt(messages=messages)

    llm_call = await llm_async_generate(llm, prompt, session)
    assert llm_call.output.content is not None
    aux_raw = llm_call.output.content

    # Strip reasoning if the policy is a thinking model.
    reasoning_delimiters = None
    llm_cfg = cfg.get("llm", None)
    if llm_cfg is not None:
        reasoning_delimiters = llm_cfg.get("reasoning_delimiters", None)
    aux_text = remove_reasoning(aux_raw, reasoning_delimiters=reasoning_delimiters)
    if not aux_text.strip():
        aux_text = aux_raw  # fall back to raw output for non-thinking models

    aux_ids = eval_tokenizer(aux_text, add_special_tokens=False).input_ids

    # ----- 4. Reward via evaluator ------------------------------------------
    # Cap (prefix + aux + suffix) at the same max_total_tokens; trim aux first
    # if the joint blows past the budget. (Trim aux from the right so that the
    # suffix portion of the joint stays intact.)
    overshoot = (len(prefix_ids) + len(aux_ids) + len(suffix_ids)) - max_total_tokens
    if overshoot > 0:
        aux_ids = aux_ids[: max(0, len(aux_ids) - overshoot)]

    # Blocking HTTP calls; offload to threads to keep the asyncio loop responsive.
    baseline_result, treatment_result = await asyncio.gather(
        asyncio.to_thread(
            evaluator_llm.get_batch_logprobs_token_ids,
            [prefix_ids],
            [suffix_ids],
        ),
        asyncio.to_thread(
            evaluator_llm.get_batch_logprobs_token_ids,
            [prefix_ids],
            [aux_ids + suffix_ids],
        ),
    )
    sum_lp_baseline = _sum_logprobs(baseline_result[0])
    sum_lp_treatment = _sum_logprobs(treatment_result[0])

    n_suffix = max(1, len(suffix_ids))
    n_joint = max(1, len(aux_ids) + len(suffix_ids))
    avg_nll_baseline = -sum_lp_baseline / n_suffix
    avg_nll_treatment = -sum_lp_treatment / n_joint
    reward = avg_nll_baseline - avg_nll_treatment

    discount_factor = float(cfg.actor.get("discount_factor", 1.0))
    if discount_factor != 1.0:
        reward = reward * (discount_factor ** llm_call.output_length_tokens)

    # ----- 5. Suffix-overlap monitoring (NOT used in reward in v0) ----------
    overlap_ratio = 0.0
    if aux_text and suffix_text_view:
        overlap_ratio = float(
            difflib.SequenceMatcher(None, aux_text, suffix_text_view, autojunk=False).ratio()
        )

    # ----- 6. Build training text and metrics -------------------------------
    trace = make_training_text(llm, llm_call)
    trace.reward = float(reward)

    metrics = Metrics(
        reward=float(reward),
        success=reward > 0.0,
        no_error=True,
        no_answer=not bool(aux_text.strip()),
        avg_nll_baseline=float(avg_nll_baseline),
        avg_nll_treatment=float(avg_nll_treatment),
        nll_delta=float(reward),
        aux_tokens=len(aux_ids),
        prefix_tokens=len(prefix_ids),
        suffix_tokens=len(suffix_ids),
        cut_offset=int(cut),
        suffix_overlap_ratio=overlap_ratio,
        skipped=False,
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=time.time() - time_start,
        dataset_name=problem.get("dataset"),
    )


def _make_skip_result(
    latency: float,
    dataset: str | None = None,
    cut_offset: int = 0,
    prefix_tokens: int = 0,
    suffix_tokens: int = 0,
) -> RolloutResult:
    """Return a zero-reward result when the row cannot be processed.

    The dispatcher requires `attempts` results per group; we return a placeholder
    rather than raising so the group still completes.
    """
    return RolloutResult(
        training_texts=[],
        metrics=Metrics(
            reward=0.0,
            success=False,
            no_error=True,
            no_answer=True,
            cut_offset=cut_offset,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            skipped=True,
        ),
        latency=latency,
        dataset_name=dataset,
    )
