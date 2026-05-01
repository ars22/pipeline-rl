"""
Verifier for AI-CoScientist (aicsi) domain.

Evaluates model-generated responses against oracle outputs using an LLM judge.
Reuses the same evaluate_idea_v1.md prompt used in offline eval (p4_evaluate.py),
parsing SIMILARITY_SCORE (0.0–1.0) as the reward signal.
"""

import asyncio
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import openai
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    pass


def _timestamp() -> str:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S,") + f"{now.microsecond // 1000:03d}"


def _infer_repo_root() -> Path:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        if (parent / "conf").is_dir():
            return parent
    return module_path.parent


_REPO_ROOT = _infer_repo_root()


def load_aicsi_prompt(prompt_name: str | os.PathLike) -> str:
    """
    Load the evaluator prompt. Tries these locations in order:
    1. Absolute path (as-is if it exists)
    2. Relative to repo root
    3. conf/evaluator_prompts/ directory in the repo
    """
    if prompt_name is None:
        raise ValueError("llm_grader.prompt_name must be set for the aicsi domain")
    prompt_str = str(prompt_name).strip()
    if not prompt_str:
        raise ValueError("llm_grader.prompt_name cannot be empty")

    # Try as absolute path
    p = Path(prompt_str)
    if p.is_absolute() and p.is_file():
        return p.read_text(encoding="utf-8")

    # Try relative to repo root
    p = (_REPO_ROOT / prompt_str).resolve()
    if p.is_file():
        return p.read_text(encoding="utf-8")

    # Try conf/evaluator_prompts/ with .md extension
    filename = Path(prompt_str).name
    if not filename.endswith(".md"):
        filename = f"{filename}.md"
    p = (_REPO_ROOT / "conf" / "evaluator_prompts" / filename).resolve()
    if p.is_file():
        return p.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Evaluator prompt '{prompt_name}' not found. "
        f"Tried absolute path, relative to repo root ({_REPO_ROOT}), "
        f"and conf/evaluator_prompts/."
    )


# ── OpenAI client ─────────────────────────────────────────────────────────────

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL environment variable")
        _openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
    return _openai_client


# ── Backend API callers ────────────────────────────────────────────────────────

def _call_nvidia_api(prompt_text: str, model: str, sampling_kwargs: dict[str, Any] | None = None) -> dict:
    import requests
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing NVIDIA_API_KEY environment variable")
    url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    kwargs = dict(sampling_kwargs) if sampling_kwargs else {}
    temperature = kwargs.pop("temperature", 1.0)
    max_tokens = kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 4096))
    kwargs.pop("reasoning", None)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=900)
    resp.raise_for_status()
    data = resp.json()
    return {"output_text": data["choices"][0]["message"]["content"], "usage": data.get("usage")}


def _call_gemini_api(prompt_text: str, model: str, sampling_kwargs: dict[str, Any] | None = None) -> dict:
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")
    genai.configure(api_key=api_key)
    kwargs = dict(sampling_kwargs) if sampling_kwargs else {}
    temperature = kwargs.pop("temperature", 1.0)
    max_tokens = kwargs.pop("max_output_tokens", kwargs.pop("max_tokens", 4096))
    kwargs.pop("reasoning", None)
    gen_model = genai.GenerativeModel(model)
    resp = gen_model.generate_content(
        prompt_text,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    usage = None
    if hasattr(resp, "usage_metadata") and resp.usage_metadata is not None:
        um = resp.usage_metadata
        usage = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "completion_tokens": getattr(um, "candidates_token_count", None),
        }
    return {"output_text": resp.text, "usage": usage}


# ── Metrics helpers ────────────────────────────────────────────────────────────

def _should_collect_metrics(collect_flag: bool | None) -> bool:
    return True if collect_flag is None else collect_flag


def _merge_metrics(base: dict, rollout: dict) -> dict:
    merged = dict(rollout)
    merged.update(base)
    return merged


def _build_rollout_metrics(success: bool, failure_causes: list[str], num_retries: int = 0) -> dict:
    metrics: dict = {}
    if success:
        metrics["verifier/rollouts/success"] = 1
    else:
        metrics["verifier/rollouts/failure"] = 1
        if failure_causes:
            unique = set(failure_causes)
            if unique == {"timeout"}:
                metrics["verifier/failures/timeout"] = 1
            elif unique == {"rate_limit"}:
                metrics["verifier/failures/rate_limit"] = 1
            elif unique == {"no_input"}:
                metrics["verifier/failures/no_input"] = 1
            elif unique == {"no_score"}:
                metrics["verifier/failures/no_score"] = 1
            else:
                metrics["verifier/failures/all_attempts_failed"] = 1
    if num_retries > 0:
        metrics["verifier/failures/num_retries"] = num_retries
    return metrics


def _extract_reasoning_from_response(response: Any) -> str:
    if isinstance(response, dict):
        return ""
    chunks = []
    for item in response.output or []:
        if getattr(item, "type", None) == "reasoning":
            for c in getattr(item, "content", []) or []:
                text = getattr(c, "text", None)
                if text:
                    chunks.append(text)
    return "\n\n".join(chunks)


def _parse_similarity_score(output_text: str) -> float | None:
    """Parse SIMILARITY_SCORE: X.XX from the evaluator output. Returns None if not found."""
    match = re.search(r"SIMILARITY_SCORE\s*:\s*([0-9]*\.?[0-9]+)", output_text)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # clamp to [0, 1]
        except ValueError:
            return None
    return None


# ── Verification result ────────────────────────────────────────────────────────

@dataclass
class AicsIVerificationResult:
    """Similarity score in [0.0, 1.0] plus runtime metrics."""
    score: float  # 0.0–1.0 (parsed from SIMILARITY_SCORE)
    metrics: dict[str, float | int] = field(default_factory=dict)
    table_entry: dict[str, str | float] | None = None


# ── Main verifier function ─────────────────────────────────────────────────────

async def verify_aicsi(
    query: str,
    oracle: str,
    generation: str,
    prompt_name: str | os.PathLike | None = None,
    model: str | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    client=None,
    timeout_seconds: int = 900,
    max_retries: int = 3,
    retry_backoff: list[int] = [15, 30, 60, 90, 120],
    log_wandb_metrics: bool | None = None,
    collect_table_entry: bool | None = None,
    backend: str = "openai",
) -> AicsIVerificationResult:
    """
    Evaluate a model-generated response against an oracle using an LLM judge.

    Uses the same evaluate_idea_v1.md prompt as offline eval (p4_evaluate.py).
    Parses SIMILARITY_SCORE (0.0–1.0) from the output as the reward.

    Args:
        query:      The scientific query (maps to PAPER_TITLE in the prompt).
        oracle:     The ground-truth oracle response (maps to GROUND_TRUTH).
        generation: The model-generated response (maps to GENERATED).
        prompt_name: Path to evaluator prompt (relative to repo root or conf/evaluator_prompts/).
        model:      Grader model name.
        backend:    One of "openai", "nvidia", "gemini".
    """
    collect_metrics = _should_collect_metrics(log_wandb_metrics)
    should_collect_table = collect_metrics if collect_table_entry is None else collect_table_entry

    if not generation.strip():
        rollout_metrics = _build_rollout_metrics(success=False, failure_causes=["no_input"], num_retries=0)
        return AicsIVerificationResult(score=0.0, metrics=_merge_metrics({}, rollout_metrics))

    if backend == "openai":
        client = client or get_openai_client()
    elif backend not in ("nvidia", "gemini"):
        raise ValueError(f"Unsupported backend: {backend!r}. Must be 'openai', 'nvidia', or 'gemini'.")

    if not model:
        raise RuntimeError("verify_aicsi requires a grader model name; set llm_grader.name in your config")

    prompt_template = load_aicsi_prompt(prompt_name)
    prompt_text = prompt_template.format(
        PAPER_TITLE=query,
        GROUND_TRUTH=oracle,
        GENERATED=generation,
    )

    api_kwargs = dict(sampling_kwargs) if sampling_kwargs else {}
    loop = asyncio.get_event_loop()

    async def _call_openai():
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(model=model, input=prompt_text, **api_kwargs),
        )

    async def _call_nvidia():
        return await loop.run_in_executor(
            None,
            lambda: _call_nvidia_api(prompt_text, model, sampling_kwargs),
        )

    async def _call_gemini():
        return await loop.run_in_executor(
            None,
            lambda: _call_gemini_api(prompt_text, model, sampling_kwargs),
        )

    _backend_callers = {"openai": _call_openai, "nvidia": _call_nvidia, "gemini": _call_gemini}

    attempt_failure_causes: list[str] = []
    num_retries = 0
    runtime_metrics: dict[str, float | int] = {}

    for attempt in range(1, max_retries + 1):
        attempt_start = time.perf_counter()
        try:
            response = await asyncio.wait_for(_backend_callers[backend](), timeout=timeout_seconds)
            latency_seconds = time.perf_counter() - attempt_start

            usage = getattr(response, "usage", None)
            if usage is None and isinstance(response, dict):
                usage = response.get("usage")
            if collect_metrics:
                runtime_metrics = {"verifier/runtime/latency_per_request": latency_seconds}
                if usage is not None:
                    out_tok = getattr(usage, "output_tokens", None) or (usage.get("output_tokens") if isinstance(usage, dict) else None)
                    in_tok = getattr(usage, "input_tokens", None) or (usage.get("input_tokens") if isinstance(usage, dict) else None)
                    if out_tok is not None:
                        runtime_metrics["verifier/runtime/output_tokens"] = out_tok
                    if in_tok is not None:
                        runtime_metrics["verifier/runtime/input_tokens"] = in_tok

            if isinstance(response, dict):
                output_text = response.get("output_text", "")
            else:
                output_text = getattr(response, "output_text", None) or ""

            similarity = _parse_similarity_score(output_text)
            if similarity is not None:
                table_entry = None
                if should_collect_table:
                    reasoning_text = _extract_reasoning_from_response(response)
                    table_entry = {
                        "prompt": prompt_text,
                        "reasoning": reasoning_text,
                        "output_text": output_text,
                        "similarity_score": str(similarity),  # str to satisfy dict[str, str | int]
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=True, failure_causes=attempt_failure_causes, num_retries=num_retries
                )
                return AicsIVerificationResult(
                    score=similarity,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )
            else:
                table_entry = None
                if should_collect_table:
                    reasoning_text = _extract_reasoning_from_response(response)
                    table_entry = {
                        "prompt": prompt_text,
                        "reasoning": reasoning_text,
                        "output_text": output_text,
                        "similarity_score": "0.0",  # str to satisfy dict[str, str | int]
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=False, failure_causes=["no_score"], num_retries=num_retries
                )
                print(f"[verify_aicsi]: {_timestamp()} - No SIMILARITY_SCORE found (attempt {attempt}) — returning 0.0")
                return AicsIVerificationResult(
                    score=0.0,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )

        except openai.RateLimitError as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("rate_limit")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_aicsi]: {_timestamp()} - Rate limit (attempt {attempt}/{max_retries}), sleeping {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

        except (asyncio.TimeoutError, TimeoutException):
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("timeout")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_aicsi]: {_timestamp()} - Timeout after {timeout_seconds}s (attempt {attempt}/{max_retries}), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("other")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_aicsi]: {_timestamp()} - Error (attempt {attempt}/{max_retries}): {e}, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print(f"[verify_aicsi]: {_timestamp()} - All {max_retries} attempts failed — returning score=0.0")
    rollout_metrics = _build_rollout_metrics(
        success=False, failure_causes=attempt_failure_causes, num_retries=num_retries
    )
    return AicsIVerificationResult(score=0.0, metrics=_merge_metrics(runtime_metrics, rollout_metrics))


# ── Environment server ─────────────────────────────────────────────────────────

class AicsIEnvironment:
    """
    FastAPI server that wraps verify_aicsi for RPC-based evaluation.
    Configure in YAML as:
        environment:
          _target_: pipelinerl.domains.aicsi.AicsIEnvironment
          model_name: ${llm_grader.name}
          sampling_kwargs: ${llm_grader.sampling_kwargs}
          prompt_name: ${llm_grader.prompt_name}
          backend: ${llm_grader.backend}
    """

    def __init__(
        self,
        model_name: str | None = None,
        sampling_kwargs: dict[str, Any] | None = None,
        use_wandb: bool | None = True,
        prompt_name: str | os.PathLike | None = None,
        backend: str = "openai",
    ):
        self.model_name = model_name
        self.sampling_kwargs = sampling_kwargs
        self.use_wandb = use_wandb
        if not prompt_name:
            raise ValueError("AicsIEnvironment requires llm_grader.prompt_name to be set")
        self.prompt_name = prompt_name
        self.backend = backend

    def launch(self, port: int):
        app = FastAPI()
        process_pool = ProcessPoolExecutor(max_workers=4)

        @app.post("/verify_answer")
        async def verify(request: dict):
            """
            Expected JSON: {"query": "...", "oracle": "...", "generation": "..."}
            Returns: {"score": 0.0–1.0}
            """
            client = get_openai_client() if self.backend == "openai" else None
            verification = await verify_aicsi(
                query=request["query"],
                oracle=request["oracle"],
                generation=request["generation"],
                prompt_name=self.prompt_name,
                client=client,
                model=self.model_name,
                sampling_kwargs=self.sampling_kwargs,
                log_wandb_metrics=self.use_wandb,
                collect_table_entry=False,
                backend=self.backend,
            )
            return JSONResponse(content={"score": verification.score})

        @app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)
