# Online Reasoning Cache (RC) Rollouts

This document describes the online rollout functionality in `rc_actor.py`.

## Quick Start

Run the RC actor with a test configuration:

```bash
python -m pipelinerl.rc_actor --config-name test_rc output_dir=/tmp/results/test_rc_actor
```

Override parameters on the command line:

```bash
python -m pipelinerl.rc_actor --config-name test_rc output_dir=/tmp/results/test_rc_actor actor.num_reasoning_steps=5
```

**Important**: Your configuration must specify `output_dir` (where results are saved). This can be set in your YAML file or via command line.

## Overview

Online RC rollouts enable iterative reasoning with summarization:
1. **Reasoning step**: Model generates reasoning for a problem
2. **Summarization step**: The reasoning is compressed into a summary
3. **State update**: Summary becomes context for next cycle
4. **Repeat**: Steps 1-3 repeat for N cycles


## Key Components

### 1. InferenceProblemState
Tracks inference progress for a single problem with the following attributes:

**State Tracking:**
- `curr_summary`: Current summarized state (initially empty, updated after each summarization)
- `curr_reasoning`: Latest reasoning output (updated after each reasoning step)
- `reasoning_turn_number`: Counter for reasoning turns (1, 2, 3, ...)
- `summarization_turn_number`: Counter for summarization turns (1, 2, 3, ...)
- `overall_cycle_step`: Counter for all steps (0, 1, 2, 3, ...)

**History Storage:**
- `reasoning_rollout_store`: All reasoning rollout results
- `summarization_rollout_store`: All summarization rollout results
- `reasoning_string_store`: Processed reasoning outputs (without think tags)
- `summarization_string_store`: Processed summaries
- `reasoning_string_complete_store`: Raw reasoning outputs (with think tags if used)
- `summarization_string_complete_store`: Raw summaries

**Prompt Generation:**
- `get_filled_reasoning_prompt()`: Fills template with `{problem}` and `{curr_summary}`
- `get_filled_summarization_prompt()`: Fills template with `{problem}`, `{existing_summary}`, and `{reasoning}`

### 2. RCActorLoop
Orchestrates the online rollout process:
- Initializes problem states from problem dict
- Puts states in queue for processing
- Manages multiple reasoning/summarization cycles
- Collects all rollouts as training data

### 3. Schedule Rollouts
Async scheduler that:
- Reads InferenceProblemState from problem queue
- Runs N reasoning/summarization cycles
- Updates state after each cycle
- Puts completed rollouts in result queue

## Workflow

```
Problem Dict
    ↓
Initialize InferenceProblemState
    ↓
Put in Problem Queue
    ↓
[Scheduler reads from queue]
    ↓
Cycle 1: Reasoning → Summarization → Update State
Cycle 2: Reasoning → Summarization → Update State
Cycle 3: Reasoning → Summarization → Update State
    ↓
Collect all rollouts
    ↓
Put results in Result Queue
    ↓
Main loop reads results and publishes
```

## Configuration

### Required Parameters

Your configuration must include:

```yaml
# Output directory (REQUIRED)
output_dir: ./runs/my_experiment

# Model to use
model_path: Qwen/Qwen3-4B-Instruct-2507
```

### Actor Configuration

Add these parameters to configure the RC actor:

```yaml
actor:
  # Number of reasoning/summarization cycles per problem
  num_reasoning_steps: 3
  
  # Number of samples to generate per problem (default: 1)
  num_samples_per_problem: 1
  
  # Whether to wrap prompts/responses with <think></think> tags
  use_think_tags: false
  
  # Prompt template for reasoning step
  # Available variables: {problem}, {curr_summary}
  reasoning_prompt_template: |
    Solve this problem step by step.
    
    Problem: {problem}
    
    Previous work: {curr_summary}
    
    Continue your reasoning:
  
  # Prompt template for summarization step
  # Available variables: {problem}, {existing_summary}, {reasoning}
  summarization_prompt_template: |
    Summarize the solution progress.
    
    Problem: {problem}
    
    Previous summary: {existing_summary}
    
    New reasoning: {reasoning}
    
    Provide a concise updated summary:
  
  # Rollout policies (functions that generate training data from model outputs)
  solution_rollout_policy: pipelinerl.domains.math.rollouts.generate_math_rollout
  summarization_rollout_policy: pipelinerl.domains.math.rollouts.generate_math_rollout
  
  # Performance settings
  rollout_workers: 1           # Number of parallel rollout workers
  llm_max_rollouts: 16         # Max concurrent rollouts per LLM
  problem_queue_size: 10       # Size of problem queue
  result_queue_size: 10        # Size of result queue
  shared_memory_entry_size: 5242880  # 5MB per entry
  throughput_window_size: 10   # Window for throughput metrics
  
  # Retry configuration for transient errors
  max_retries: 3
  retry_base_delay: 1.0
```

### LLM Configuration

```yaml
llm:
  parameters:
    temperature: 1.0      # Sampling temperature
    max_tokens: 16384     # Max tokens per generation
    top_p: 0.95          # Nucleus sampling

test_llm:
  parameters:
    temperature: 1.0
    max_tokens: 16384
    top_p: 1.0
```

### Other Settings

```yaml
# Random seed
seed: 42

# Streams backend (where to write rollout data)
streams:
  backend: files

# Disable WandB for testing
wandb:
  use_wandb: false

# Training attempts (for online RC, typically use 1)
attempts: 1

# Evaluation frequency (0 = disabled)
eval_every_n_versions: 0

# Debug mode
debug:
  mode: true
```

See `conf/test_rc.yaml` for a complete example configuration.

## Metadata

Each `TrainingText` in the rollout results includes the following metadata, which is **set immediately** when the rollout is created (not at the end):

```python
{
    "model_version": 0,           # Model version used for this rollout
    "rollout_index": 0,           # Which attempt this was (typically 0 for attempts=1)
    "cycle_step": 2,              # Overall cycle step index (0, 1, 2, 3, 4, 5, ...)
    "turn_type": "reasoning",     # Type of turn: "reasoning" or "summarization"
    "turn_number": 2,             # Which turn of that type (1, 2, 3, ...)
}
```

The turn numbers are tracked in `InferenceProblemState`:
- `reasoning_turn_number`: Incremented each time a reasoning rollout is completed
- `summarization_turn_number`: Incremented each time a summarization rollout is completed
- `overall_cycle_step`: Incremented after each rollout (reasoning or summarization)

### Example with `num_reasoning_steps=3`:

| cycle_step | turn_type      | turn_number | Description              |
|------------|----------------|-------------|--------------------------|
| 0          | reasoning      | 1           | First reasoning turn     |
| 1          | summarization  | 1           | First summarization turn |
| 2          | reasoning      | 2           | Second reasoning turn    |
| 3          | summarization  | 2           | Second summarization turn|
| 4          | reasoning      | 3           | Third reasoning turn     |
| 5          | summarization  | 3           | Third summarization turn |

This metadata allows you to:
- Filter rollouts by turn type (e.g., only use reasoning turns for training)
- Weight different turns differently (e.g., higher weight for later turns)
- Analyze performance by turn number
- Track progression through the reasoning/summarization cycles

