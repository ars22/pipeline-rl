import time
import requests
import asyncio
import importlib
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


async def verify_answer_rpc_genrm(
    cfg: DictConfig,
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    prediction: str,
    gold: str,
    strict: bool = True,
    max_prediction_length: int = 1000,
    # prompt, solution and eot_token are needed for genrm
    prompt: str = "",
    solution: str = "",
    eot_token: str = "",
    prompt_template_path: str = "",
    return_score: bool = True,
):
    """
    Verify the answer using the verifier API.
    """
    if prompt_template_path:
        prompt_template = importlib.import_module(prompt_template_path)
        gen_rm_system_prompt = prompt_template.SYSTEM_PROMPT
        gen_rm_user_prompt_template = prompt_template.USER_PROMPT
            
    model = cfg.genrm_llm.model_path
    json = {        
        "prediction": prediction,
        "gold": gold,
        "strict": strict,
        "max_prediction_length": max_prediction_length,
        "prompt": prompt,
        "solution": solution,
        "eot_token": eot_token,
        "gen_rm_system_prompt": gen_rm_system_prompt if prompt_template_path else None,
        "gen_rm_user_prompt_template": gen_rm_user_prompt_template if prompt_template_path else None,
        "model": model,
    }
    async with session.post(
        f"http://{host}:{port}/verify_answer",
        json=json,
    ) as response:
        if response.status == 200:
            data = await response.json()
            if return_score:
                return data["answer_status"], data.get("genrm_score", None)
            else:
                return data["answer_status"]
        else:
            logger.error(f"Error verifying answer: {response.status}")
            logger.error(f"Response: {await response.text()}")
            raise ValueError("Error verifying answer")


async def call_genrm(session: aiohttp.ClientSession, 
    url: str, user_prompt: str, system_prompt: str, model: str) -> float:
    """
    Call the GenRM to get a score for the prediction.
    """
    # TODO: Use a proper template
    full_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json={
                "model": model,
                "messages": full_prompt,
                "max_tokens": 4096,
                "temperature": 0.0,
                # we always disable thinking for genrm
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        ) as response:
            if response.status == 200:
                data = await response.json()
                text = data["choices"][0]["message"]["content"]                
                try:
                    if "Score:" in text:
                        score_str = text.split("Score:")[1].lstrip().split("\n")[0]
                    else:
                        score_str = text.split("Score")[1].lstrip().split("\n")[0]
                    score_str = score_str.replace("**", "").replace("*", "")[0]
                    score = int(score_str[0])
                    if score not in [1, 2, 3, 4, 5]:
                        score = None
                except Exception as e:
                    score = None
            else:
                logger.error(f"GenRM error: {response.status} {await response.text()}")
                score = None
    except Exception as e:
        logger.error(f"GenRM exception: {e}")
        score = None
    return score


import random

class MathEnvironment:

    def launch(self, port: int, debug_mode: bool = False):
        """
        Serve the verification API using FastAPI.
        """
        app = FastAPI()
        # Create a process pool with 4 workers
        if debug_mode:
            from concurrent.futures import ThreadPoolExecutor
            executor_cls = ThreadPoolExecutor
            max_workers = 1
        else:
            executor_cls = ProcessPoolExecutor
            max_workers = 4

        with executor_cls(max_workers=max_workers) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                if debug_mode:
                    print(f"Add traces here..")
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


class GenRMMathEnvironment:  
    def __init__(self, prompt_template_path: str, *args, **kwargs):
        self.prompt_template_path = prompt_template_path
    
    def launch(self, port: int, debug_mode: bool = False, genrm_urls: list[str] | None = None):
        """
        Serve the verification API using FastAPI with GenRM support.
        """
        app = FastAPI()        
        if debug_mode:
            from concurrent.futures import ThreadPoolExecutor
            executor_cls = ThreadPoolExecutor
            max_workers = 1
        else:
            executor_cls = ProcessPoolExecutor
            max_workers = 4

        with executor_cls(max_workers=max_workers) as process_pool:
            @app.post("/verify_answer")
            async def verify(request: dict):
                system_prompt = request.get("gen_rm_system_prompt", "")
                user_prompt_template = request.get("gen_rm_user_prompt_template", "")
                prediction = request["prediction"]
                gold = request["gold"]
                strict = request["strict"]
                max_prediction_length = request["max_prediction_length"]
                prompt = request.get("prompt", "")
                solution = request.get("solution", "")
                eot_token = request.get("eot_token", "")

                user_prompt = user_prompt_template.format(
                    problem=prompt, reasoning_trace=prediction.replace(eot_token, ""), solution=solution)
                model = request.get("model", "")
                # normal verification for outcome scores
                if debug_mode:
                    # run in main thread to avoid signal issues with math_verify
                    answer_status = verify_answer(prediction, gold, strict, max_prediction_length)
                else:
                    loop = asyncio.get_event_loop()
                    answer_status = await loop.run_in_executor(
                        process_pool, partial(verify_answer, prediction, gold, strict, max_prediction_length)
                    )

                assert genrm_urls, "genrm_urls is not set"                
                genrm_url = random.choice(genrm_urls)
                async with aiohttp.ClientSession() as session:
                    genrm_score = await call_genrm(
                        session, genrm_url, user_prompt, 
                        system_prompt, model)

                return JSONResponse(content={"answer_status": answer_status, "genrm_score": genrm_score})

            @app.get("/health")
            async def health():
                return JSONResponse(content={"status": "ok"})

            uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=60)