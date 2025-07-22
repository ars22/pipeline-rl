"""Evaluation script for TIR (Tool Integrated Reasoning) domain."""

import logging
import time
from typing import Optional

import numpy as np
from datasets import load_dataset
from termcolor import colored
from tqdm import tqdm
import wandb

from tapeagents.llms import TrainableLLM

from .agent import TIRMathAgent, extract_result_value, solve_task, AnswerAction
from .environment import TIRMathEnvironment, MCPPythonEnvironment
from .prompts import PromptRegistry

# Set logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_tir_model(
    num_samples: int = 100,
    temperature: float = 0.2,
    dataset_name: str = "gsm8k",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    log_failures: bool = True,
    use_mcp: bool = True,
    prompt_type: str = "default",
    model_path: str = "/mnt/llmd/base_models/AI-MO-NuminaMath-7B-TIR",
    base_url: str = "http://localhost:8080"
):
    """
    Evaluate the TIR model's performance on math problems.
    
    Args:
        num_samples: Number of samples to evaluate
        temperature: Sampling temperature for the model
        dataset_name: Dataset to evaluate on (gsm8k, math, etc.)
        wandb_project: W&B project name (if None, no logging to wandb)
        wandb_run_name: W&B run name 
        log_failures: Whether to log detailed failure analysis to wandb
        use_mcp: Whether to use MCP Python execution environment
        prompt_type: Type of prompt to use (default, advanced)
        model_path: Path to the model
        base_url: Base URL for the LLM server
        
    Returns:
        Dictionary with accuracy and detailed results.
    """
    # Initialize wandb if project specified
    if wandb_project:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"tir-{dataset_name}-eval-t{temperature}-n{num_samples}",
            config={
                "dataset": dataset_name.upper(),
                "num_samples": num_samples,
                "temperature": temperature,
                "model": model_path.split("/")[-1] if "/" in model_path else model_path,
                "framework": "TapeAgents-TIR",
                "execution_env": "MCP-Python" if use_mcp else "Container-Python",
                "prompt_type": prompt_type,
                "max_iterations": 1,
            }
        )
    
    # Load dataset
    if dataset_name.lower() == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
    elif dataset_name.lower() == "math":
        dataset = load_dataset("hendrycks/competition_math", "main", split="test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    samples = [s for s in dataset][:num_samples]
    logger.info(f"Evaluating on {len(samples)} samples from {dataset_name.upper()} test set using TIR")

    # Initialize model and agent
    llm = TrainableLLM(
        base_url=base_url,
        model_name=model_path,
        tokenizer_name=model_path,
        parameters=dict(
            temperature=temperature,
            max_tokens=512
        ),
    )
    
    # Get the appropriate system prompt
    system_prompt = PromptRegistry.get_prompt(prompt_type)
    
    agent = TIRMathAgent.create(
        llm=llm, 
        max_prompt_length=1024,
        system_prompt=system_prompt
    )
    
    # Use TIR environment
    if use_mcp:
        env = MCPPythonEnvironment()
        logger.info("Using MCP Python environment for TIR")
    else:
        env = TIRMathEnvironment(use_mcp=False)
        logger.info("Using container/restricted Python environment for TIR")
    
    # Track results
    results = []
    correct = 0
    errors = 0
    
    # Track detailed metrics for wandb
    no_answer_count = 0
    wrong_answer_count = 0
    execution_errors = 0
    
    # Start timing
    start_time = time.time()
    sample_times = []
    
    for i, sample in enumerate(tqdm(samples)):
        try:
            # Time individual sample processing
            sample_start_time = time.time()
            
            # Prepare sample with expected value
            sample = extract_result_value(sample)
            
            # Use the solve_task function from TIR agent
            tape = solve_task(agent, env, sample)
            
            # Check if the task was solved correctly
            is_solved = tape.metadata.result.get("solved", False)
            
            # Count steps in the tape
            num_steps = len(tape.steps)
            
            # Extract the answer from the last AnswerAction in the tape
            answer_step = None
            for step in reversed(tape.steps):
                if isinstance(step, AnswerAction):
                    answer_step = step
                    break
            
            if answer_step is None:
                logger.warning(colored(f"No answer found for sample {i}", "yellow"))
                no_answer_count += 1
                errors += 1
                results.append({
                    "question": sample["question"],
                    "expected": sample["value"],
                    "predicted": None,
                    "correct": False,
                    "solved": False,
                    "error": "No answer produced",
                    "sample_id": i,
                    "num_steps": num_steps
                })
                
                # Log no-answer cases immediately to wandb for debugging
                if wandb_project:
                    wandb.log({"no_answer_case": i + 1, "no_answer_total": no_answer_count})
                continue
                
            # Compare results
            predicted_value = answer_step.value
            expected_value = sample["value"]
            
            # Check if values match (with small tolerance for floating point)
            if predicted_value is not None and expected_value is not None:
                is_correct = abs(float(predicted_value) - float(expected_value)) < 1e-6
            else:
                is_correct = False
            
            if is_correct:
                correct += 1
                logger.debug(colored(f"Correct answer for sample {i}", "green"))
            else:
                wrong_answer_count += 1
                logger.debug(colored(f"Wrong answer for sample {i}. Expected {expected_value}, got {predicted_value}", "red"))
            
            # Record timing for this sample
            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            sample_times.append(sample_duration)
            
            results.append({
                "question": sample["question"],
                "expected": expected_value,
                "predicted": predicted_value,
                "correct": is_correct,
                "solved": is_solved,
                "error": None,
                "sample_id": i,
                "processing_time": sample_duration,
                "num_steps": num_steps
            })
            
            # Log progress to wandb every 10 samples
            if wandb_project and (i + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_sample = elapsed_time / (i + 1)
                current_step_counts = [r["num_steps"] for r in results[:i+1]]
                current_avg_steps = np.mean(current_step_counts) if current_step_counts else 0
                wandb.log({
                    "samples_processed": i + 1,
                    "current_accuracy": correct / (i + 1),
                    "current_no_answer_rate": no_answer_count / (i + 1),
                    "current_error_rate": errors / (i + 1),
                    "elapsed_time_minutes": elapsed_time / 60,
                    "avg_time_per_sample_seconds": avg_time_per_sample,
                    "estimated_total_time_minutes": (avg_time_per_sample * len(samples)) / 60,
                    "current_avg_steps_per_sample": current_avg_steps
                })
            
        except Exception as e:
            logger.error(colored(f"Error processing sample {i}: {str(e)}", "red"))
            execution_errors += 1
            errors += 1
            
            # Record timing even for failed samples
            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            sample_times.append(sample_duration)
            
            # Set num_steps to 0 for failed samples (no tape created)
            num_steps = 0
            
            results.append({
                "question": sample["question"],
                "expected": sample.get("value", None),
                "predicted": None,
                "correct": False,
                "solved": False,
                "error": str(e),
                "sample_id": i,
                "processing_time": sample_duration,
                "num_steps": num_steps
            })
    
    # Calculate metrics and timing
    total_time = time.time() - start_time
    avg_time_per_sample = np.mean(sample_times) if sample_times else 0
    median_time_per_sample = np.median(sample_times) if sample_times else 0
    
    # Calculate step statistics
    step_counts = [r["num_steps"] for r in results]
    avg_steps_per_sample = np.mean(step_counts) if step_counts else 0
    median_steps_per_sample = np.median(step_counts) if step_counts else 0
    max_steps = max(step_counts) if step_counts else 0
    min_steps = min(step_counts) if step_counts else 0
    
    accuracy = correct / len(samples) if len(samples) > 0 else 0
    error_rate = errors / len(samples) if len(samples) > 0 else 0
    solved_rate = sum(1 for r in results if r["solved"]) / len(samples) if len(samples) > 0 else 0
    no_answer_rate = no_answer_count / len(samples) if len(samples) > 0 else 0
    wrong_answer_rate = wrong_answer_count / len(samples) if len(samples) > 0 else 0
    execution_error_rate = execution_errors / len(samples) if len(samples) > 0 else 0
    
    logger.info(f"\n{dataset_name.upper()} TIR Evaluation Results:")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(colored(f"Accuracy: {accuracy:.2%} ({correct}/{len(samples)})", "green"))
    logger.info(colored(f"Solved rate: {solved_rate:.2%}", "blue"))
    logger.info(f"Correct: {correct}")
    logger.info(colored(f"Errors: {errors} ({error_rate:.2%})", "red"))
    logger.info(f"Token usage: {llm.token_count if hasattr(llm, 'token_count') else 'N/A'}")
    logger.info(colored(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)", "cyan"))
    logger.info(colored(f"Average time per sample: {avg_time_per_sample:.2f}s", "cyan"))
    logger.info(colored(f"Median time per sample: {median_time_per_sample:.2f}s", "cyan"))
    logger.info(colored(f"Average steps per sample: {avg_steps_per_sample:.1f}", "magenta"))
    logger.info(colored(f"Median steps per sample: {median_steps_per_sample:.1f}", "magenta"))
    logger.info(colored(f"Step range: {min_steps}-{max_steps}", "magenta"))
    
    # Log final results to wandb
    if wandb_project:
        final_metrics = {
            "final_accuracy": accuracy,
            "final_solved_rate": solved_rate,
            "final_error_rate": error_rate,
            "no_answer_rate": no_answer_rate,
            "wrong_answer_rate": wrong_answer_rate,
            "execution_error_rate": execution_error_rate,
            "total_samples": len(samples),
            "correct_answers": correct,
            "no_answer_cases": no_answer_count,
            "wrong_answers": wrong_answer_count,
            "execution_errors": execution_errors,
            "token_usage": getattr(llm, 'token_count', None),
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "avg_time_per_sample_seconds": avg_time_per_sample,
            "median_time_per_sample_seconds": median_time_per_sample,
            "samples_per_minute": len(samples) / (total_time / 60) if total_time > 0 else 0,
            "avg_steps_per_sample": avg_steps_per_sample,
            "median_steps_per_sample": median_steps_per_sample,
            "max_steps": max_steps,
            "min_steps": min_steps
        }
        wandb.log(final_metrics)
        
        # Log failure examples as a table
        if log_failures:
            failures = [r for r in results if not r["correct"]]
            failure_data = []
            for failure in failures[:20]:  # Log first 20 failures
                failure_data.append([
                    failure["sample_id"],
                    failure["question"][:150] + "..." if len(failure["question"]) > 150 else failure["question"],
                    failure["expected"],
                    failure["predicted"],
                    failure["error"] or "Wrong calculation"
                ])
            
            if failure_data:
                failure_table = wandb.Table(
                    columns=["Sample ID", "Question", "Expected", "Predicted", "Error Type"],
                    data=failure_data
                )
                wandb.log({"failure_examples": failure_table})
        
        # Create accuracy over time chart
        progress_data = []
        running_correct = 0
        for i, result in enumerate(results):
            if result["correct"]:
                running_correct += 1
            progress_data.append([i + 1, running_correct / (i + 1)])
        
        progress_table = wandb.Table(
            columns=["Sample", "Accuracy"],
            data=progress_data
        )
        wandb.log({
            "accuracy_over_time": wandb.plot.line(
                progress_table, "Sample", "Accuracy", 
                title=f"{dataset_name.upper()} TIR Accuracy Over Time"
            )
        })
        
        wandb.finish()
    
    return {
        "accuracy": accuracy,
        "solved_rate": solved_rate,
        "error_rate": error_rate,
        "no_answer_rate": no_answer_rate,
        "wrong_answer_rate": wrong_answer_rate,
        "execution_error_rate": execution_error_rate,
        "total_samples": len(samples),
        "correct": correct,
        "errors": errors,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "avg_time_per_sample": avg_time_per_sample,
        "median_time_per_sample": median_time_per_sample,
        "samples_per_minute": len(samples) / (total_time / 60) if total_time > 0 else 0,
        "avg_steps_per_sample": avg_steps_per_sample,
        "median_steps_per_sample": median_steps_per_sample,
        "max_steps": max_steps,
        "min_steps": min_steps,
        "detailed_results": results
    }


def analyze_failures(results: dict, num_examples: int = 5):
    """Analyze and print example failures for debugging."""
    failures = [r for r in results["detailed_results"] if not r["correct"]]
    
    print(f"\nTIR Failure Analysis ({len(failures)} total failures):")
    print("=" * 80)
    
    # Group failures by type
    no_answer = [f for f in failures if f["predicted"] is None]
    wrong_answer = [f for f in failures if f["predicted"] is not None]
    
    print(f"No answer produced: {len(no_answer)}")
    print(f"Wrong answer: {len(wrong_answer)}")
    
    print(f"\nExample Failures (showing first {num_examples}):")
    for i, failure in enumerate(failures[:num_examples]):
        print(f"\nFailure {i+1}:")
        print(f"Sample ID: {failure.get('sample_id', 'N/A')}")
        print(f"Question: {failure['question']}")
        print(f"Expected: {failure['expected']}")
        print(f"Predicted: {failure['predicted']}")
        print(f"Solved: {failure['solved']}")
        if failure['error']:
            print(f"Error: {failure['error']}")
        print("-" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TIR evaluation on math datasets")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"], help="Dataset to evaluate on")
    parser.add_argument("--wandb-project", type=str, help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, help="Wandb run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--no-mcp", action="store_true", help="Don't use MCP Python execution")
    parser.add_argument("--prompt-type", type=str, default="default", choices=["default", "advanced"], help="Prompt type to use")
    parser.add_argument("--model-path", type=str, default="/mnt/llmd/base_models/AI-MO-NuminaMath-7B-TIR", help="Model path")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080", help="Base URL for LLM server")
    
    args = parser.parse_args()
    
    # Set up wandb configuration
    wandb_project = None if args.no_wandb else (args.wandb_project or f"tir-{args.dataset}-eval")
    
    logger.info(f"Starting TIR evaluation:")
    logger.info(f"  Dataset: {args.dataset.upper()}")
    logger.info(f"  Number of samples: {args.num_samples}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Prompt type: {args.prompt_type}")
    logger.info(f"  Use MCP: {not args.no_mcp}")
    logger.info(f"  Wandb project: {wandb_project or 'None (disabled)'}")
    
    # Run TIR evaluation
    results = evaluate_tir_model(
        num_samples=args.num_samples,
        temperature=args.temperature,
        dataset_name=args.dataset,
        wandb_project=wandb_project,
        wandb_run_name=args.wandb_run_name,
        log_failures=True,
        use_mcp=not args.no_mcp,
        prompt_type=args.prompt_type,
        model_path=args.model_path,
        base_url=args.base_url
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TIR {args.dataset.upper()} EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total_samples']})")
    print(f"No answer rate: {results['no_answer_rate']:.2%}")
    print(f"Wrong answer rate: {results['wrong_answer_rate']:.2%}")
    print(f"Execution error rate: {results['execution_error_rate']:.2%}")
    print(f"Total time: {results['total_time_minutes']:.1f} minutes")
    print(f"Average time per sample: {results['avg_time_per_sample']:.2f}s")
    print(f"Throughput: {results['samples_per_minute']:.1f} samples/minute")
    print(f"Average steps per sample: {results['avg_steps_per_sample']:.1f}")
    print(f"Step range: {results['min_steps']}-{results['max_steps']}")
    
    if wandb_project:
        print(f"\nResults logged to wandb project: {wandb_project}")
    
    # Analyze failures
    analyze_failures(results)