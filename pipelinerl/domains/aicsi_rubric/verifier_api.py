"""
Rubric-based verifier for AI-CoScientist (aicsi_rubric) domain.

Evaluates model-generated responses against a set of binary rubric criteria
using an LLM judge. Reward = n_yes / n_total (normalized to [0, 1]).

Uses the evaluate_rubric.md prompt from the v3 eval pipeline, parsing
---SCORE--- delimited JSON judgments.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List

# Only match {ALL_CAPS_WITH_UNDERSCORES} as template placeholders — avoids
# conflicts with JSON braces in the evaluate_rubric.md example output.
_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")

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

SCORE_DELIMITER = "---SCORE---"
MAX_RESPONSE_CHARS = 12000


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


def load_rubric_prompt(prompt_name: str | os.PathLike) -> str:
    """
    Load the rubric evaluator prompt. Tries these locations in order:
    1. Absolute path (as-is if it exists)
    2. Relative to repo root
    3. conf/evaluator_prompts/ directory in the repo
    """
    if not prompt_name:
        raise ValueError("llm_grader.prompt_name must be set for the aicsi_rubric domain")
    prompt_str = str(prompt_name).strip()

    p = Path(prompt_str)
    if p.is_absolute() and p.is_file():
        return p.read_text(encoding="utf-8")

    p = (_REPO_ROOT / prompt_str).resolve()
    if p.is_file():
        return p.read_text(encoding="utf-8")

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


# ── OpenAI client ──────────────────────────────────────────────────────────────

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


# ── Rubric helpers ─────────────────────────────────────────────────────────────

def format_rubric(criteria: List[dict]) -> str:
    """Format rubric criteria list into numbered lines for the judge prompt."""
    lines = []
    for c in criteria:
        cid = c.get("criterion_id", "")
        aspect = c.get("aspect", "")
        criterion = c.get("criterion", "")
        lines.append(f"{cid}. [{aspect}] {criterion}")
    return "\n".join(lines)


def truncate(text: str, max_chars: int = MAX_RESPONSE_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated ...]"


def parse_rubric_scores(raw_text: str) -> List[dict]:
    """Parse ---SCORE--- delimited JSON judgment objects from LLM output."""
    scores = []
    parts = raw_text.split(SCORE_DELIMITER)
    for part in parts:
        part = part.strip()
        if not part:
            continue

        json_str = re.sub(r"```(?:json)?\s*", "", part).strip()

        try:
            s = json.loads(json_str)
            if isinstance(s, dict) and "judgment" in s:
                scores.append(s)
                continue
        except json.JSONDecodeError:
            pass

        # Fallback: brace-matching extraction
        brace_depth = 0
        start = None
        for i, ch in enumerate(json_str):
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start is not None:
                    try:
                        s = json.loads(json_str[start:i + 1])
                        if isinstance(s, dict) and "judgment" in s:
                            scores.append(s)
                    except json.JSONDecodeError:
                        pass
                    start = None

    return scores


def compute_rubric_score(scores: List[dict], n_criteria: int) -> float:
    """Compute normalized rubric score: n_yes / n_criteria."""
    for s in scores:
        j = str(s.get("judgment", "")).lower().strip()
        s["judgment"] = "yes" if j in ("yes", "y", "true", "1") else "no"
    n_yes = sum(1 for s in scores if s["judgment"] == "yes")
    if n_criteria <= 0:
        return 0.0
    return max(0.0, min(1.0, n_yes / n_criteria))


# ── Metrics helpers ────────────────────────────────────────────────────────────

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


def _merge_metrics(base: dict, rollout: dict) -> dict:
    merged = dict(rollout)
    merged.update(base)
    return merged


# ── Verification result ────────────────────────────────────────────────────────

@dataclass
class RubricVerificationResult:
    """Rubric score in [0.0, 1.0] plus runtime metrics."""
    score: float
    metrics: dict[str, float | int] = field(default_factory=dict)
    table_entry: dict[str, str | float] | None = None


# ── Main verifier function ─────────────────────────────────────────────────────

async def verify_rubric(
    query: str,
    rubric: List[dict],
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
) -> RubricVerificationResult:
    """
    Evaluate a model-generated response against rubric criteria using an LLM judge.

    Uses evaluate_rubric.md prompt. Parses ---SCORE--- delimited judgments and
    returns score = n_yes / n_criteria as the reward signal.

    Args:
        query:      The scientific query (maps to QUERY in the prompt).
        rubric:     List of criterion dicts (maps to RUBRIC in the prompt).
        generation: The model-generated response (maps to CANDIDATE_RESPONSE).
        prompt_name: Path to evaluator prompt.
        model:      Grader model name.
        backend:    One of "openai", "nvidia", "gemini".
    """
    collect_metrics = True if log_wandb_metrics is None else log_wandb_metrics
    should_collect_table = collect_metrics if collect_table_entry is None else collect_table_entry

    if not generation.strip():
        rollout_metrics = _build_rollout_metrics(success=False, failure_causes=["no_input"], num_retries=0)
        return RubricVerificationResult(score=0.0, metrics=_merge_metrics({}, rollout_metrics))

    if not rubric:
        rollout_metrics = _build_rollout_metrics(success=False, failure_causes=["no_input"], num_retries=0)
        return RubricVerificationResult(score=0.0, metrics=_merge_metrics({}, rollout_metrics))

    if backend == "openai":
        client = client or get_openai_client()
    elif backend not in ("nvidia", "gemini"):
        raise ValueError(f"Unsupported backend: {backend!r}. Must be 'openai', 'nvidia', or 'gemini'.")

    if not model:
        raise RuntimeError("verify_rubric requires a grader model name; set llm_grader.name in config")

    prompt_template = load_rubric_prompt(prompt_name)
    rubric_text = format_rubric(rubric)

    def _fill(template: str, **kwargs) -> str:
        def _replace(m):
            return str(kwargs.get(m.group(1), m.group(0)))
        return _PLACEHOLDER_RE.sub(_replace, template)

    prompt_text = _fill(
        prompt_template,
        QUERY=query,
        CANDIDATE_RESPONSE=truncate(generation),
        RUBRIC=rubric_text,
    )

    n_criteria = len(rubric)
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

            scores = parse_rubric_scores(output_text)
            if scores:
                score = compute_rubric_score(scores, n_criteria)
                table_entry = None
                if should_collect_table:
                    table_entry = {
                        "prompt": prompt_text,
                        "output_text": output_text,
                        "rubric_score": str(score),
                        "n_yes": str(sum(1 for s in scores if s.get("judgment") == "yes")),
                        "n_criteria": str(n_criteria),
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=True, failure_causes=attempt_failure_causes, num_retries=num_retries
                )
                return RubricVerificationResult(
                    score=score,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )
            else:
                # No parseable scores — return 0
                table_entry = None
                if should_collect_table:
                    table_entry = {
                        "prompt": prompt_text,
                        "output_text": output_text,
                        "rubric_score": "0.0",
                        "n_yes": "0",
                        "n_criteria": str(n_criteria),
                    }
                rollout_metrics = _build_rollout_metrics(
                    success=False, failure_causes=["no_score"], num_retries=num_retries
                )
                print(f"[verify_rubric]: {_timestamp()} - No ---SCORE--- judgments found (attempt {attempt}) — returning 0.0")
                return RubricVerificationResult(
                    score=0.0,
                    metrics=_merge_metrics(runtime_metrics, rollout_metrics),
                    table_entry=table_entry,
                )

        except openai.RateLimitError as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("rate_limit")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_rubric]: {_timestamp()} - Rate limit (attempt {attempt}/{max_retries}), sleeping {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

        except (asyncio.TimeoutError, TimeoutException):
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("timeout")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_rubric]: {_timestamp()} - Timeout after {timeout_seconds}s (attempt {attempt}/{max_retries}), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            attempt_failure_causes.append("other")
            if attempt < max_retries:
                num_retries += 1
            print(f"[verify_rubric]: {_timestamp()} - Error (attempt {attempt}/{max_retries}): {e}, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print(f"[verify_rubric]: {_timestamp()} - All {max_retries} attempts failed — returning score=0.0")
    rollout_metrics = _build_rollout_metrics(
        success=False, failure_causes=attempt_failure_causes, num_retries=num_retries
    )
    return RubricVerificationResult(score=0.0, metrics=_merge_metrics(runtime_metrics, rollout_metrics))


# ── Environment server ─────────────────────────────────────────────────────────

class AicsIRubricEnvironment:
    """
    FastAPI server wrapping verify_rubric for RPC-based evaluation.

    Configure in YAML as:
        environment:
          _target_: pipelinerl.domains.aicsi_rubric.AicsIRubricEnvironment
          model_name: ${llm_grader.name}
          sampling_kwargs: ${llm_grader.sampling_kwargs}
          prompt_name: ${llm_grader.prompt_name}
          backend: ${llm_grader.backend}

    POST /verify_answer expects:
        {"query": "...", "rubric": [...], "generation": "..."}
    Returns:
        {"score": 0.0–1.0}
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
            raise ValueError("AicsIRubricEnvironment requires llm_grader.prompt_name to be set")
        self.prompt_name = prompt_name
        self.backend = backend

    def launch(self, port: int):
        app = FastAPI()

        @app.post("/verify_answer")
        async def verify(request: dict):
            """
            Expected JSON:
                {"query": "...", "rubric": [...], "generation": "..."}
            rubric can also be a JSON string (will be parsed automatically).
            Returns: {"score": 0.0–1.0}
            """
            rubric = request.get("rubric", [])
            if isinstance(rubric, str):
                try:
                    rubric = json.loads(rubric)
                except json.JSONDecodeError:
                    rubric = []

            client = get_openai_client() if self.backend == "openai" else None
            verification = await verify_rubric(
                query=request["query"],
                rubric=rubric,
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
