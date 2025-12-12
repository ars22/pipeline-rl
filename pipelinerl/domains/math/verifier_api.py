import time
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import signal
import argparse
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import math_verify  # Ensure math_verify is installed

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from functools import partial

import pipelinerl.countdown_utils

logging.basicConfig(
    level=logging.DEBUG,  # Or INFO, WARNING, etc.
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to console
    ],
)


logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


class UnparsableException(Exception):
    pass


class NoAnswerException(Exception):
    pass


class EmptyBoxedException(Exception):
    pass


@contextmanager
def timeout(seconds=1):
    def timeout_handler(signum, frame):
        raise TimeoutException("Computation timed out")

    # Set the timeout handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield  # This is the key addition - context managers must yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def verify_answer(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    """
    Checks if a predicted answer matches a gold (correct) answer by making a request to the math_verify package.

    Args:
        prediction (str): The predicted answer to validate
        gold (str): The gold (correct) answer to compare against
        strict (bool): Whether to enforce strict comparison mode.
        - In strict mode: Variables matter and sets are not comparable with tuples
        - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        url (str): URL of the validation service endpoint

    Returns:
        str: The status of the answer, which can be one of the following:
        - "correct": The prediction is correct
        - "wrong": The prediction is incorrect
        - "no_answer": The prediction is empty
        - "unparsable": The prediction cannot be parsed

    """
    if prediction.startswith("countdown"):
        return verify_countdown(prediction, gold)
    else:
        return verify_math(prediction, gold, strict=strict, max_prediction_length=max_prediction_length)


def verify_math(prediction: str, gold: str, strict: bool = True, max_prediction_length: int = 1000) -> str:
    try:
        # Input Sanitization / Validation (very important)
        if not isinstance(prediction, str) or not isinstance(gold, str):
            raise ValueError("Prediction and gold must be strings")

        boxed_start = prediction.rfind("\\boxed{")

        if boxed_start < 0:
            raise NoAnswerException()

        boxed_prediction = prediction[boxed_start:]
        if "\\boxed{}" in boxed_prediction:
            raise EmptyBoxedException()

        if len(boxed_prediction) > max_prediction_length:
            raise UnparsableException()

        gold_parsed = math_verify.parse(gold)
        boxed_prediction_parsed = math_verify.parse(boxed_prediction)
        if not boxed_prediction_parsed:
            raise ValueError("Failed to parse prediction.")

        with timeout(1):
            equivalent = math_verify.verify(gold_parsed, boxed_prediction_parsed, strict=strict, timeout_seconds=1)
        if equivalent:
            answer_status = "correct"
        else:
            answer_status = "wrong"

    except Exception as e:
        match e:
            case NoAnswerException():
                answer_status = "no_answer"
            case _:
                answer_status = "unparsable"
    return answer_status


def verify_countdown(prediction: str, gold: str) -> str:
    target = eval(gold.split("-")[1])
    numbers = eval(gold.split("-")[2])

    equation = pipelinerl.countdown_utils.extract_solution(solution_str=prediction)

    if equation is None:
        return "no_answer"

    format_correct = pipelinerl.countdown_utils.validate_format(prediction)
    if not format_correct:
        return "unparsable"

    # Validate equation uses correct numbers
    if not pipelinerl.countdown_utils.validate_equation(equation, numbers):
        return "wrong"

    # Evaluate equation
    try:
        result = pipelinerl.countdown_utils.evaluate_equation(equation)
        if result is None:
            return "wrong"

        if abs(result - target) < 1e-5:  # Account for floating point precision
            return "correct"
        else:
            return "wrong"
    except Exception as _:
        return "wrong"


async def verify_answer_rpc(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    gold: str,
    strict: bool = True,
    max_prediction_length: int = 1000,
):
    """
    Verify the answer using the verifier API.
    """
    json = {
        "prediction": prediction,
        "gold": gold,
        "strict": strict,
        "max_prediction_length": max_prediction_length,
    }
    async with session.post(
        f"http://{host}:{port}/verify_answer",
        json=json,
    ) as response:
        if response.status == 200:
            data = await response.json()
            return data["answer_status"]
        else:
            logger.error(f"Error verifying answer: {response.status}")
            logger.error(f"Response: {await response.text()}")
            raise ValueError("Error verifying answer")


class MathEnvironment:

    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 4 workers
        with ProcessPoolExecutor(max_workers=4) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                prediction = request["prediction"]
                gold = request["gold"]
                strict = request["strict"]
                max_prediction_length = request["max_prediction_length"]

                # Run verification in the process pool to avoid blocking the main thread
                loop = asyncio.get_event_loop()
                answer_status = await loop.run_in_executor(
                    process_pool, partial(verify_answer, prediction, gold, strict, max_prediction_length)
                )
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)


import re
import asyncio
import os
import openai
from datasets import load_dataset

PROOF_EVALUATOR_PROMPT = """
You are an **expert mathematics proof grader**. Your role is to evaluate the correctness,
rigor, and completeness of a model-generated proof for a given problem using a provided
**reference solution** and **grading schema**.

---

### INPUT COMPONENTS
You will receive four sections:
1. **Problem Statement** — the math problem to be solved.
2. **Reference Solution** — a correct, authoritative solution.
3. **Marking Scheme (Schema)** — a JSON list of checkpoints (`title`, `desc`, `points`).
4. **Proof Solution** — the model-generated proof.

---

### YOUR TASK
Analyze the **Proof Solution** against the **Problem**, **Reference Solution**, and **Schema**,
determine correspondence with rubric checkpoints, and assign an integer score **0–7**.

---

### OUTPUT FORMAT
Respond *only* in XML:

<assessment>DETAILED_EVALUATION_TEXT</assessment>
<errors>
  1. description of first issue,
  2. description of second issue,
  ...
</errors>
<score>INTEGER</score>

---

### INPUT DATA

**Problem Statement**
{problem}

**Reference Solution**
{human_solution}

**Marking Scheme**
{marking_scheme}

**Proof Solution**
{solution}
"""

_openai_client = None

@dataclass
class ProofVerificationResult:
    score: int
    metrics: dict[str, float | int] = field(default_factory=dict)
    table_entry: dict[str, str | int] | None = None


def _extract_reasoning_from_response(response: Any) -> str:
    """
    Extract reasoning content from an OpenAI Responses API Response object.

    The Response object has an `output` list containing output items.
    For reasoning models, this includes items with type="reasoning" that have
    a `content` list of text objects.

    See: https://platform.openai.com/docs/api-reference/responses/object
    """
    reasoning_chunks: list[str] = []
    for item in response.output or []:
        if getattr(item, "type", None) == "reasoning":
            for content_item in getattr(item, "content", []) or []:
                text = getattr(content_item, "text", None)
                if text:
                    reasoning_chunks.append(text)
    return "\n\n".join(reasoning_chunks)


def _should_collect_metrics(collect_flag: bool | None) -> bool:
    if collect_flag is None:
        return True
    return collect_flag


def _merge_metrics(
    base_metrics: dict[str, float | int],
    rollout_metrics: dict[str, float | int],
) -> dict[str, float | int]:
    metrics = dict(rollout_metrics)
    metrics.update(base_metrics)
    return metrics


def _build_rollout_metrics(success: bool, failure_causes: list[str], num_retries: int = 0) -> dict[str, int]:
    metrics: dict[str, int] = {}
    if success:
        metrics["verifier/rollouts/success"] = 1
    else:
        metrics["verifier/rollouts/failure"] = 1
        failure_metric_recorded = False
        if failure_causes:
            unique_causes = set(failure_causes)
            if unique_causes == {"timeout"}:
                metrics["verifier/failures/timeout"] = 1
                failure_metric_recorded = True
            elif unique_causes == {"rate_limit"}:
                metrics["verifier/failures/rate_limit"] = 1
                failure_metric_recorded = True
            elif unique_causes == {"no_input"}:
                metrics["verifier/failures/no_input"] = 1
                failure_metric_recorded = True
            elif unique_causes == {"no_score_tag"}:
                metrics["verifier/failures/no_score_tag"] = 1
                failure_metric_recorded = True
        if not failure_metric_recorded:
            metrics["verifier/failures/all_attempts_failed"] = 1

    if num_retries > 0:
        metrics["verifier/failures/num_retries"] = num_retries
    return metrics


def get_openai_client():
    """
    Lazily initialize and cache a OpenAI API client using OpenAI SDK interface.
    Requires OPENAI_API_KEY to be set in environment.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL environment variable")
        _openai_client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    return _openai_client 

# =================================================
# Proof evaluator: calls OpenAI-compatible endpoint
# =================================================
async def verify_proof(
    problem: str,
    ref_solution: str,
    schema: str,
    generation: str,
    model: str | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    client=None,
    timeout_seconds: int = 900,
    max_retries: int = 3,
    retry_backoff: list[int] = [15, 30, 60, 90, 120],
    log_wandb_metrics: bool | None = None,
) -> ProofVerificationResult:
    """
    Evaluate a model-generated proof via Groq GPR model.
    Returns a ProofVerificationResult that includes the integer score [0–7] and optional runtime metrics.
    Retries up to `max_retries` times if the OpenAI-compatible endpoint fails or hits rate limits.
    """

    collect_metrics = _should_collect_metrics(log_wandb_metrics)

    if len(generation.strip()) == 0:
        rollout_metrics = _build_rollout_metrics(success=False, failure_causes=["no_input"], num_retries=0)
        return ProofVerificationResult(
            score=0,
            metrics=_merge_metrics({}, rollout_metrics),
        )

    client = client or get_openai_client()

    prompt_text = PROOF_EVALUATOR_PROMPT.format(
        problem=problem,
        human_solution=ref_solution,
        marking_scheme=schema,
        solution=generation,
    )
    if not model:
        raise RuntimeError("verify_proof requires a grader model name; pass via cfg.llm_grader.name")
    api_kwargs = dict(sampling_kwargs) if sampling_kwargs else {}

    loop = asyncio.get_event_loop()

    # TODO: add support for chat completions API for other graders
    async def _call_openai():
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model=model,
                input=prompt_text,
                **api_kwargs,
            ),
        )

    attempt_failure_causes: list[str] = []
    num_retries = 0
    runtime_metrics: dict[str, float | int] = {}

    for attempt in range(1, max_retries + 1):
        attempt_start = time.perf_counter()
        try:
            response = await asyncio.wait_for(_call_openai(), timeout=timeout_seconds)
            latency_seconds = time.perf_counter() - attempt_start
            usage = getattr(response, "usage", None)
            output_tokens = None
            input_tokens = None
            if usage is not None:
                output_tokens = getattr(usage, "output_tokens", None)
                input_tokens = getattr(usage, "input_tokens", None)
                if output_tokens is None and isinstance(usage, dict):
                    output_tokens = usage.get("output_tokens")
                if input_tokens is None and isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens")
            if collect_metrics:
                runtime_metrics = {"verifier/runtime/latency_per_request": latency_seconds}
                if output_tokens is not None:
                    runtime_metrics["verifier/runtime/output_tokens"] = output_tokens
                if input_tokens is not None:
                    runtime_metrics["verifier/runtime/input_tokens"] = input_tokens
            output_text = getattr(response, "output_text", None) or ""
            match = re.search(r"<score>(\d+)</score>", output_text)
            if match:
                score = int(match.group(1))
                table_entry = None
                if collect_metrics:
                    reasoning_text = _extract_reasoning_from_response(response)
                    table_entry = {
                        "prompt": prompt_text,
                        "reasoning": reasoning_text,
                        "output_text": output_text,
                        "score": score,
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=True,
                    failure_causes=attempt_failure_causes,
                    num_retries=num_retries,
                )
                return ProofVerificationResult(
                    score=score,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )
            else:
                table_entry = None
                if collect_metrics:
                    reasoning_text = _extract_reasoning_from_response(response)
                    table_entry = {
                        "prompt": prompt_text,
                        "reasoning": reasoning_text,
                        "output_text": output_text,
                        "score": 0,
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=False,
                    failure_causes=["no_score_tag"],
                    num_retries=num_retries,
                )
                print(f"[verify_proof] No <score> tag found (attempt {attempt}) — returning 0")
                return ProofVerificationResult(
                    score=0,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )

        except openai.RateLimitError as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("rate_limit")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_proof] Rate limit hit (attempt {attempt}/{max_retries}), sleeping {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

        except (asyncio.TimeoutError, TimeoutException):
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("timeout")
            if attempt < max_retries:
                num_retries += 1
            print(
                f"[verify_proof] Timeout after {timeout_seconds}s (attempt {attempt}/{max_retries}), "
                f"retrying in {wait_time}s..."
            )
            await asyncio.sleep(wait_time)

        except Exception as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("other")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_proof] Error on attempt {attempt}/{max_retries}: {e}, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print(f"[verify_proof] All {max_retries} attempts failed — returning score=0")
    rollout_metrics = _build_rollout_metrics(
        success=False,
        failure_causes=attempt_failure_causes,
        num_retries=num_retries,
    )
    return ProofVerificationResult(
        score=0,
        metrics=_merge_metrics(runtime_metrics, rollout_metrics),
    )

class MathProofEnvironment:
    def __init__(
        self,
        model_name: str | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
        use_wandb: bool | None = True,
    ):
        self.model_name = model_name
        self.sampling_kwargs = sampling_kwargs
        self.use_wandb = use_wandb

    def launch(self, port: int):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        process_pool = ProcessPoolExecutor(max_workers=4)

        @app.post("/verify_answer")
        async def verify(request: dict):
            """
            Evaluate a proof-based problem.
            Expected JSON:
            {
                "problem": "...",
                "ref_solution": "...",
                "schema": "...",
                "generation": "..."
            }
            """
            problem = request["problem"]
            ref_solution = request["ref_solution"]
            schema = request["schema"]
            generation = request["generation"]

            client = get_openai_client()
            verification = await verify_proof(
                problem=problem,
                ref_solution=ref_solution,
                schema=schema,
                generation=generation,
                client=client,
                model=self.model_name,
                sampling_kwargs=self.sampling_kwargs,
                log_wandb_metrics=self.use_wandb,
            )
            return JSONResponse(content={"score": verification.score})

        @app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)

# =====================================================================
# General llm-based math evaluator: calls an OpenAI-compatible endpoint
# =====================================================================

MATH_LLM_SYSTEM_PROMPT = """You are an expert mathematician and a meticulous AI reasoning evaluator. Your task is to assess the quality of a "Generated Reasoning" trace.
Specifically, you must judge how likely it is that a model, *after* thinking through the "Generated Reasoning," would then produce the *exact* "Reference Solution" provided.
You are evaluating the **logical and causal link** between the reasoning trace and the final answer."""

MATH_LLM_USER_PROMPT = """### Math Problem

{problem}

### Generated Reasoning

{reasoning_trace}

### Reference Solution

{reference_solution}

### Instructions
First, provide a step-by-step analysis of the connection between the "Generated Reasoning" and the "Reference Solution." In your analysis, consider the following:
* **Correctness & Alignment:** Is the reasoning trace mathematically sound, and does its logical conclusion *perfectly match* the "Reference Solution"?
* **Sufficiency:** Does the reasoning provide all the necessary steps and logic to arrive at the answer? Or does it stop short, requiring an additional logical leap?
* **Contradiction:** Is there anything in the reasoning that would contradict the "Reference Solution" or lead to a *different* answer, even if part of the reasoning is correct?
After your analysis, provide a single numerical score on the 5-point Likert scale defined below.

**Likert Scale:**
* **1 (Very Unlikely):** The reasoning is incorrect, stops far short, or actively leads to a completely different answer.
* **2 (Unlikely):** The reasoning has significant gaps, flaws, or is too vague, making the "Reference Solution" a surprising or illogical next step.
* **3 (Neutral/Possible):** The reasoning is generally on the right track but is incomplete or contains minor errors. One *could* arrive at the answer, but it's not a direct or guaranteed consequence.
* **4 (Likely):** The reasoning is correct and provides a strong, clear path to the answer, with only trivial gaps (if any).
* **5 (Very Likely):** The reasoning is sound, complete, and directly and unambiguously implies the "Reference Solution" as its final conclusion.

---

Please follow this output format:
Reasoning: [Your detailed analysis goes here.]

Score: [Provide the single numerical score: 1, 2, 3, 4, or 5.]"""


@dataclass
class MathLLMVerificationResult:
    score: int  # 1-5
    metrics: dict[str, float | int] = field(default_factory=dict)
    table_entry: dict[str, str | int] | None = None


async def verify_math_llm(
    problem: str,
    reasoning_trace: str,
    reference_solution: str,
    model: str | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    client=None,
    timeout_seconds: int = 300,
    max_retries: int = 3,
    retry_backoff: list[int] = [15, 30, 60, 90, 120],
    log_wandb_metrics: bool | None = None,
) -> MathLLMVerificationResult:
    """
    Evaluate model-generated reasoning via LLM judge.
    Returns a MathLLMVerificationResult with score [1-5].
    """
    collect_metrics = _should_collect_metrics(log_wandb_metrics)

    if len(reasoning_trace.strip()) == 0:
        rollout_metrics = _build_rollout_metrics(success=False, failure_causes=["no_input"], num_retries=0)
        return MathLLMVerificationResult(
            score=1,  # Minimum score for empty reasoning
            metrics=_merge_metrics({}, rollout_metrics),
        )

    client = client or get_openai_client()

    user_prompt = MATH_LLM_USER_PROMPT.format(
        problem=problem,
        reasoning_trace=reasoning_trace,
        reference_solution=reference_solution,
    )
    
    if not model:
        raise RuntimeError("verify_math_llm requires a grader model name; pass via cfg.llm_grader.name")
    
    api_kwargs = dict(sampling_kwargs) if sampling_kwargs else {}
    # Remove reasoning-specific kwargs that don't apply to chat completions
    api_kwargs.pop("reasoning", None)
    api_kwargs.pop("max_output_tokens", None)
    
    loop = asyncio.get_event_loop()

    # Use Chat Completions API for standard LLM graders
    async def _call_openai():
        return await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MATH_LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **api_kwargs,
            ),
        )

    attempt_failure_causes: list[str] = []
    num_retries = 0
    runtime_metrics: dict[str, float | int] = {}

    for attempt in range(1, max_retries + 1):
        attempt_start = time.perf_counter()
        try:
            response = await asyncio.wait_for(_call_openai(), timeout=timeout_seconds)
            latency_seconds = time.perf_counter() - attempt_start
            
            # Extract usage metrics
            usage = getattr(response, "usage", None)
            output_tokens = None
            input_tokens = None
            if usage is not None:
                output_tokens = getattr(usage, "completion_tokens", None)
                input_tokens = getattr(usage, "prompt_tokens", None)
            
            if collect_metrics:
                runtime_metrics = {"verifier/runtime/latency_per_request": latency_seconds}
                if output_tokens is not None:
                    runtime_metrics["verifier/runtime/output_tokens"] = output_tokens
                if input_tokens is not None:
                    runtime_metrics["verifier/runtime/input_tokens"] = input_tokens
                        
            output_text = response.choices[0].message.content or ""
            # match = re.search(r"Score:\s*(\d)", output_text)
            # extract score from output_text
            try:
                if "Score:" in output_text:
                    score_str = output_text.split("Score:")[1].lstrip().split("\n")[0]
                elif "Score" in output_text:
                    score_str = output_text.split("Score")[1].lstrip().split("\n")[0]
                else:
                    raise ValueError(f"Score not found in response: {output_text}")
                
                score_str = score_str.replace("*", "").lstrip()[0]
                raw_score = int(score_str[0])
                # @TODO: currently default to smooth strategy, change it later
                score = (raw_score - 1) / 4
                
                table_entry = None
                if collect_metrics:
                    table_entry = {
                        "system_prompt": MATH_LLM_SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "output_text": output_text,
                        "score": score,
                        "raw_score": raw_score,
                    }
                
                rollout_metrics = _build_rollout_metrics(
                    success=True,
                    failure_causes=attempt_failure_causes,
                    num_retries=num_retries,
                )
                return MathLLMVerificationResult(
                    score=score,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )
            except Exception as e:                
                # if no score found, we disgard the full trace
                table_entry = None
                if collect_metrics:
                    table_entry = {
                        "system_prompt": MATH_LLM_SYSTEM_PROMPT,
                        "user_prompt": user_prompt,
                        "output_text": output_text,
                        "score": 1,
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=False,
                    failure_causes=["no_score_tag"],
                    num_retries=num_retries,
                )
                print(f"[verify_math_llm] No 'Score: X' found (attempt {attempt}) — returning 1")
                return MathLLMVerificationResult(
                    score=1,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )

        except openai.RateLimitError as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("rate_limit")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_math_llm] Rate limit hit (attempt {attempt}/{max_retries}), sleeping {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

        except (asyncio.TimeoutError, TimeoutException):
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("timeout")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_math_llm] Timeout (attempt {attempt}/{max_retries}), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("other")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_math_llm] Error on attempt {attempt}/{max_retries}: {e}, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print(f"[verify_math_llm] All {max_retries} attempts failed — returning score=1")
    rollout_metrics = _build_rollout_metrics(
        success=False,
        failure_causes=attempt_failure_causes,
        num_retries=num_retries,
    )
    return MathLLMVerificationResult(
        score=1,
        metrics=_merge_metrics(runtime_metrics, rollout_metrics),
    )


class MathLLMEnvironment:
    """Environment that uses LLM-as-judge for math reasoning evaluation."""
    
    def __init__(
        self,
        model_name: str | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
        use_wandb: bool | None = True,
    ):
        self.model_name = model_name
        self.sampling_kwargs = sampling_kwargs
        self.use_wandb = use_wandb

    def launch(self, port: int):
        """Serve the LLM verification API using FastAPI."""
        app = FastAPI()

        @app.post("/verify_answer")
        async def verify(request: dict):
            """
            Evaluate reasoning quality via LLM judge.
            Expected JSON:
            {
                "problem": "...",
                "reasoning_trace": "...",
                "reference_solution": "..."
            }
            """
            problem = request["problem"]
            reasoning_trace = request["reasoning_trace"]
            reference_solution = request["reference_solution"]

            client = get_openai_client()
            verification = await verify_math_llm(
                problem=problem,
                reasoning_trace=reasoning_trace,
                reference_solution=reference_solution,
                client=client,
                model=self.model_name,
                sampling_kwargs=self.sampling_kwargs,
                log_wandb_metrics=self.use_wandb,
            )
            return JSONResponse(content={"score": verification.score})

        @app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)


def main():
    parser = argparse.ArgumentParser(description="Run proof verifier locally for debugging.")
    parser.add_argument("--model", required=True, help="Fully qualified grader model name (e.g. openai/gpt-oss-20b)")
    parser.add_argument(
        "--sampling-kwargs",
        default=None,
        help="JSON dict of sampling params forwarded to the grader (e.g. '{\"temperature\": 0.8}')",
    )
    args = parser.parse_args()
    sampling_kwargs = json.loads(args.sampling_kwargs) if args.sampling_kwargs else None

    dataset = load_dataset("hf-imo-colab/olympiads-proof-schema", split="train")
    data = dataset[1]
    problem = data["problem"]
    ref_solution = data["solution"]
    schema = data["schema_0"]
    prediction = data["solution"]
    for i in range(10):
        verification = asyncio.run(
            verify_proof(
                problem,
                ref_solution,
                schema,
                prediction,
                model=args.model,
                sampling_kwargs=sampling_kwargs,
            )
        )
        print(f"Score: {verification.score}")

if __name__ == "__main__":
    main()



