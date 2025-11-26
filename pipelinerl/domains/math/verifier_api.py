import time
import requests
import asyncio
from concurrent.futures import ProcessPoolExecutor
import aiohttp
import uvicorn
import logging
import signal
from contextlib import contextmanager

from omegaconf import DictConfig
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

# ===========================================================
# Proof evaluator: calls Groq 120B grader via responses.create
# ===========================================================
async def verify_proof(
    problem: str,
    ref_solution: str,
    schema: str,
    generation: str,
    client=None,
    timeout_seconds: int = 500,
    max_retries: int = 3,
    retry_backoff: list[int] = [10, 30, 60],
) -> int:
    """
    Evaluate a model-generated proof via Groq GPR model.
    Returns an integer score [0–7].
    Retries up to `max_retries` times if Groq API fails or hits rate limits.
    """
    client = client or get_openai_client()

    prompt_text = PROOF_EVALUATOR_PROMPT.format(
        problem=problem,
        human_solution=ref_solution,
        marking_scheme=schema,
        solution=generation,
    )

    loop = asyncio.get_event_loop()

    async def _call_openai():
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(
                model="openai/gpt-oss-120b", # TODO: make this configurable
                input=prompt_text,
                reasoning={"effort": "high"},
                temperature=0.0,
                max_output_tokens=16384,
            ),
            # lambda: client.chat.completions.create(
            #     model="openai/gpt-oss-120b",
            #     messages=[
            #         {"role": "user", "content": prompt_text}
            #     ],
            #     temperature=0.0,
            #     max_tokens=16384,
            # ),
        )

    for attempt in range(1, max_retries + 1):
        try:
            response = await asyncio.wait_for(_call_openai(), timeout=timeout_seconds)
            output_text = getattr(response, "output_text", None) or ""
            # output_text = response.choices[0].delta.content
            match = re.search(r"<score>(\d+)</score>", output_text)
            if match:
                return int(match.group(1))
            else:
                print(f"[verify_proof] No <score> tag found (attempt {attempt}) — returning 0")
                return 0

        except openai.RateLimitError as e:
            # handle Groq 429 rate limit
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            print(f"[verify_proof] Rate limit hit (attempt {attempt}/{max_retries}), sleeping {wait_time}s: {e}")
            await asyncio.sleep(wait_time)

        except asyncio.TimeoutError:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            print(f"[verify_proof] Timeout after {timeout_seconds}s (attempt {attempt}/{max_retries}), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

        except Exception as e:
            wait_time = retry_backoff[min(attempt - 1, len(retry_backoff) - 1)]
            print(f"[verify_proof] Error on attempt {attempt}/{max_retries}: {e}, retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)

    print(f"[verify_proof] All {max_retries} attempts failed — returning score=0")
    return 0

class MathProofEnvironment:
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
            score = await verify_proof(
                problem=problem,
                ref_solution=ref_solution,
                schema=schema,
                generation=generation,
                client=client,
            )
            return JSONResponse(content={"score": score})

        @app.get("/health")
        async def health():
            return JSONResponse(content={"status": "ok"})

        uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)

def main():
    dataset = load_dataset("hf-imo-colab/olympiads-proof-schema", split="train")
    data = dataset[1]
    problem = data["problem"]
    ref_solution = data["solution"]
    schema = data["schema_0"]
    prediction = data["solution"]
    for i in range(10):
        score = asyncio.run(verify_proof(problem, ref_solution, schema, prediction))
        print(f"Score: {score}")

if __name__ == "__main__":
    main()