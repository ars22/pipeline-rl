# Online Reasoning Cache (RC) Rollouts

This document describes the online rollout functionality in `rc_actor.py`, inspired by the [verl-stable reasoning cache implementation](https://raw.githubusercontent.com/IanYHWu/verl-stable/refs/heads/reasoning_cache_offline/projects/reasoning_cache/rollouts.py).

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

