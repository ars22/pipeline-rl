# Online Reasoning Cache (RC) Rollouts

This document describes the online rollout functionality in `rc_actor.py`.

## Quick Start

Run the RC actor with:

```bash
python -m pipelinerl.rc_actor --config-name base
```

Or use a custom config:

```bash
python -m pipelinerl.rc_actor --config-name test_rc
```

Override parameters:

```bash
python -m pipelinerl.rc_actor --config-name base actor.num_reasoning_steps=5
```

## Overview

Online RC rollouts enable iterative reasoning with summarization:
1. **Reasoning step**: Model generates reasoning for a problem
2. **Summarization step**: The reasoning is compressed into a summary
3. **State update**: Summary becomes context for next cycle
4. **Repeat**: Steps 1-3 repeat for N cycles


## Key Components

### 1. InferenceProblemState
Tracks inference progress for a single problem:
- **Current state**: `curr_summary`, `curr_reasoning`
- **History**: Stores all reasoning and summarization rollouts
- **Prompts**: Generates prompts with current state as context

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

Add these parameters to the actor configuration:

```yaml
actor:
  # Number of reasoning/summarization cycles per problem
  num_reasoning_steps: 3
  
  # Number of samples to generate per problem
  num_samples_per_problem: 1
  
  # Whether to use <think></think> tags
  use_think_tags: false
  
  # Prompt template for reasoning step
  # Variables: {problem}, {curr_summary}
  reasoning_prompt_template: |
    Problem: {problem}
    
    Current summary: {curr_summary}
    
    Continue reasoning:
  
  # Prompt template for summarization step
  # Variables: {problem}, {existing_summary}, {reasoning}
  summarization_prompt_template: |
    Problem: {problem}
    
    Existing summary: {existing_summary}
    
    New reasoning: {reasoning}
    
    Provide an updated summary:
  
  # Rollout policies
  solution_rollout_policy: pipelinerl.domains.math.rollouts.generate_math_rollout
  summarization_rollout_policy: pipelinerl.domains.math.rollouts.generate_math_rollout
```

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

