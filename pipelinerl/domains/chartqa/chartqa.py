import time
import base64
import io
import logging
from typing import Dict, Any

import aiohttp
from omegaconf import DictConfig
from pydantic import BaseModel
from PIL import Image

from pipelinerl.rollouts import RolloutResult
from tapeagents.core import Prompt
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.async_llm import llm_async_generate, make_training_text
from .evaluation import evaluate_answer

logger = logging.getLogger(__name__)


class ChartQARewardTable(BaseModel):
    wrong_answer_not_finished: float
    wrong_answer_finished: float
    no_answer_not_finished: float
    no_answer_finished: float
    unparsable_not_finished: float
    unparsable_finished: float
    correct_answer_not_finished: float
    correct_answer_finished: float


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string with optimization."""
    # Resize large images for efficiency
    max_size = (1024, 1024)
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image = image.copy()
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffered = io.BytesIO()
    # Always use PNG for compatibility with Qwen2.5-VL
    if image.mode in ('RGBA', 'LA'):
        image.save(buffered, format="PNG", optimize=True)
    else:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG", optimize=True)
    
    img_str = base64.b64encode(buffered.getvalue()).decode()
    # Use the format that Qwen2.5-VL expects
    return f"data:image;base64,{img_str}"


def create_multimodal_message(image: Image.Image, question: str) -> Dict[str, Any]:
    """Create a multimodal message with image and text."""
    image_base64 = encode_image_to_base64(image)
    
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": image_base64
                }
            },
            {
                "type": "text", 
                "text": question
            }
        ]
    }


async def generate_chartqa_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    """Generate a rollout for ChartQA domain."""
    logger.info(f"Starting ChartQA rollout for problem: {problem.get('id', 'unknown')}")
    logger.info(f"Problem keys: {list(problem.keys())}")
    logger.info(f"Question: {str(problem.get('question', 'N/A'))[:100]}...")
    logger.info(f"Expected answer: {problem.get('answer', 'N/A')}")
    logger.info(f"Image type: {type(problem.get('image', 'N/A'))}")
    
    messages = []
    
    # Add system prompt if specified
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
        logger.info(f"Added system prompt: {str(cfg.actor.system_prompt)[:50]}...")
    
    # Create the multimodal user message with chart image and question
    try:
        question_text = cfg.actor.task_template.format(question=problem["question"])
        logger.info(f"Formatted question: {str(question_text)[:100]}...")
        
        multimodal_message = create_multimodal_message(problem["image"], question_text)
        messages.append(multimodal_message)
        logger.info(f"Created multimodal message with {len(multimodal_message['content'])} content parts")
        
        # Log the structure for debugging
        for i, content in enumerate(multimodal_message['content']):
            if content['type'] == 'image_url':
                logger.info(f"Content {i}: type={content['type']}, url_prefix={content['image_url']['url'][:50]}...")
            else:
                logger.info(f"Content {i}: type={content['type']}, text={content.get('text', '')[:50]}...")
        
    except Exception as e:
        logger.error(f"Error creating multimodal message: {e}")
        raise
    
    prompt = Prompt(messages=messages)
    logger.info(f"Created prompt with {len(messages)} messages")

    time_start = time.time()
    try:
        llm_call = await llm_async_generate(llm, prompt, session)
        latency = time.time() - time_start
        logger.info(f"LLM call completed in {latency:.2f}s")
        logger.info(f"Generated response: {str(llm_call.output.content)[:200]}...")
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        raise

    assert llm_call.output.content is not None
    rewards = ChartQARewardTable(**dict(cfg.rewards))
    discount_factor = cfg.actor.discount_factor
    logger.info(f"Using discount factor: {discount_factor}")

    # Evaluate the answer using our custom evaluation logic
    try:
        answer_status = evaluate_answer(llm_call.output.content, problem["answer"])
        logger.info(f"Answer evaluation: {answer_status}")
        logger.info(f"Predicted: '{str(llm_call.output.content).strip()}'")
        logger.info(f"Expected: '{problem['answer']}'")
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        answer_status = "unparsable"

    try:
        trace = make_training_text(llm, llm_call)
        # Check if the generation is finished (ended with EOS token)
        finished = 1 if trace.input_ids[-1] == llm.tokenizer.eos_token_id else 0
        logger.info(f"Generation finished: {bool(finished)}")
    except Exception as e:
        logger.error(f"Error creating training text: {e}")
        raise

    # Determine reward based on answer status and finished state
    try:
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
        trace.reward = reward
        logger.info(f"Final reward: {reward} (status: {answer_status}, finished: {finished})")
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
        raise

    metrics = {
        "reward": reward,
        "success": answer_status == "correct",
        "no_error": answer_status != "unparsable",
        "no_answer": answer_status == "no_answer",
        "overflow": 0 if finished else 1,
    }
    logger.info(f"Rollout metrics: {metrics}")

    result = RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency, 
        dataset_name=problem.get("dataset"),
        prompt_tokens=[llm_call.prompt_length_tokens],
        output_tokens=[llm_call.output_length_tokens],
    )
    logger.info(f"Completed ChartQA rollout successfully")
    return result