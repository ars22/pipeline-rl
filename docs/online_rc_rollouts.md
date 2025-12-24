# Online Reasoning Cache (RC) Rollouts

This document describes the online rollout functionality in `rc_actor.py`, inspired by the [verl-stable reasoning cache implementation](https://raw.githubusercontent.com/IanYHWu/verl-stable/refs/heads/reasoning_cache_offline/projects/reasoning_cache/rollouts.py).

## Overview

Online RC rollouts enable iterative reasoning with summarization:
1. **Reasoning step**: Model generates reasoning for a problem
2. **Summarization step**: The reasoning is compressed into a summary
3. **State update**: Summary becomes context for next cycle
4. **Repeat**: Steps 1-3 repeat for N cycles

This approach helps with:
- **Long reasoning chains**: Break complex problems into manageable chunks
- **Memory efficiency**: Compress reasoning history via summaries
- **Iterative refinement**: Each cycle builds on previous work

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

Add these parameters to your actor configuration:

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

## Detailed Workflow

### 1. Initialize Problem State

```python
state = InferenceProblemState(
    problem_text=problem['task'],
    problem_id=problem['id'],
    answer=problem['answer'],
    dataset_name=problem['dataset'],
    sample_id=0,
    reasoning_prompt_template=reasoning_prompt_template,
    summarization_prompt_template=summarization_prompt_template,
    use_think_tags=False,
    starting_step=0
)
```

### 2. Put State in Queue

```python
self.problem_queue.put(init_problem_state, block=False)
```

### 3. Scheduler Processes State

For each reasoning step (N times):

```python
# 1. Generate reasoning based on current summary
reasoning_prompt = state.get_filled_reasoning_prompt()
# Prompt: "Problem: {X}\nCurrent summary: {Y}\nContinue:"

reasoning_problem = {
    "task": reasoning_prompt,
    "answer": state.answer,
    "dataset": state.dataset_name,
    "id": state.problem_id,
}

solution_result = await solution_rollout_policy(cfg, llm, reasoning_problem, session)

# Extract reasoning text from result
reasoning_text = extract_text_from_rollout(solution_result)

# Update state with new reasoning
state.update_reasoning([solution_result], reasoning_text)

# 2. Summarize the reasoning
summarization_prompt = state.get_filled_summarization_prompt()
# Prompt: "Problem: {X}\nExisting summary: {Y}\nNew reasoning: {Z}\nSummarize:"

summarization_problem = {
    "task": summarization_prompt,
    "answer": state.answer,
    "dataset": state.dataset_name,
    "id": state.problem_id,
}

summary_result = await summarization_rollout_policy(cfg, llm, summarization_problem, session)

# Extract summary text from result
summary_text = extract_text_from_rollout(summary_result)

# Update state with new summary (becomes context for next cycle)
state.update_summarization([summary_result], summary_text)

# Collect rollouts for training
all_rollout_results.append(solution_result)
all_rollout_results.append(summary_result)
```

### 4. Collect and Return Results

After all N cycles:

```python
# Keep each rollout separate - don't combine them
# Set metadata on each rollout result
for cycle_step_index, result in enumerate(all_rollout_results):
    result.model_version = model_version
    result.group_id = full_group_id
    
    for sample in result.training_texts:
        sample.metadata["model_version"] = model_version
        sample.metadata["rollout_index"] = rollout_index
        sample.metadata["cycle_step"] = cycle_step_index
        sample.group_id = full_group_id

# Add all rollouts to group (list of separate RolloutResults)
group_rollouts[group_id].extend(all_rollout_results)
group_attempts_completed[group_id] += 1

# When all attempts complete, put all rollouts in result queue
if group_attempts_completed[group_id] == attempts:
    result_queue.put(group_rollouts[group_id])
```

With `num_reasoning_steps=3` and `attempts=1`:
- Puts a list of **6 RolloutResults** (3 reasoning + 3 summarization)
- Each RolloutResult is kept separate with its own metadata

### 5. Main Loop Processes Results

```python
# Read from result queue
rollout_results = self.result_queue.get(block=False)

# Publish training data
for r in rollout_results:
    for text in r.training_texts:
        all_text_dumps.append(text.model_dump())
data_stream_writer.write(all_text_dumps)
```

## Example: Math Problem

### Configuration

```yaml
actor:
  num_reasoning_steps: 3
  reasoning_prompt_template: |
    Solve this math problem step by step.
    
    Problem: {problem}
    
    Previous work: {curr_summary}
    
    Continue:
  
  summarization_prompt_template: |
    Summarize the solution so far.
    
    Problem: {problem}
    Previous summary: {existing_summary}
    New work: {reasoning}
    
    Updated summary:
```

### Execution

**Problem:** "What is the derivative of x²?"

**Cycle 1:**
- **Reasoning**: "To find the derivative, I'll use the power rule..."
- **Summary**: "Started solving using power rule"

**Cycle 2:**
- **Context**: Summary from Cycle 1
- **Reasoning**: "Applying the power rule: d/dx(x²) = 2x^(2-1)..."
- **Summary**: "Applied power rule: d/dx(x²) = 2x¹"

**Cycle 3:**
- **Context**: Summary from Cycle 2
- **Reasoning**: "Simplifying: 2x¹ = 2x. Therefore the answer is 2x."
- **Summary**: "Final answer: 2x"

**Training Data**: All 6 rollouts (3 reasoning + 3 summarization) are written as **separate RolloutResults**:
1. RolloutResult from Reasoning Cycle 1 (cycle_step=0)
2. RolloutResult from Summarization Cycle 1 (cycle_step=1)
3. RolloutResult from Reasoning Cycle 2 (cycle_step=2)
4. RolloutResult from Summarization Cycle 2 (cycle_step=3)
5. RolloutResult from Reasoning Cycle 3 (cycle_step=4)
6. RolloutResult from Summarization Cycle 3 (cycle_step=5)

Each RolloutResult contains its own training_texts and can be processed independently.

## Metrics

Track the progression through cycles:
- Number of reasoning steps completed
- Number of summarization steps completed
- Total rollouts per problem: `2 × num_reasoning_steps`
- Average prompt/output tokens per cycle

## Comparison to Standard Actor

| Feature | Standard Actor | RC Actor |
|---------|---------------|----------|
| **Input** | Problem dict | Problem dict → InferenceProblemState |
| **Cycles** | Single rollout | Multiple reasoning/summarization cycles |
| **State** | Stateless | Stateful (maintains summary between cycles) |
| **Output** | 1 rollout per problem | 2N rollouts per problem (N cycles) |
| **Context** | Static problem | Growing summary context |

## Best Practices

### 1. Choose Appropriate num_reasoning_steps

- **Too few**: May not reach complete solution
- **Too many**: Wastes computation if solution found early
- **Recommendation**: Start with 3, adjust based on problem complexity

### 2. Design Clear Prompts

**Reasoning prompt:**
- Should encourage forward progress
- Include current summary for context
- Be specific about what to continue

**Summarization prompt:**
- Should compress without losing key information
- Ask for concise updates
- Focus on progress made

### 3. Monitor Throughput

Each reasoning step doubles the rollouts vs standard actor:
- `standard_actor`: 1 rollout per problem
- `rc_actor (N=3)`: 6 rollouts per problem (3 reasoning + 3 summarization)

**Adjust accordingly:**
- Reduce `num_reasoning_steps` for higher throughput
- Increase `llm_max_rollouts` to maintain speed
- Add more LLMs to parallelize

### 4. Check Summary Quality

Monitor if summaries are helpful:
- Are they capturing key progress?
- Are they too verbose or too terse?
- Do later cycles build on them effectively?

## Implementation Details

### State Updates

```python
class InferenceProblemState:
    def update_reasoning(self, rollouts, response_string):
        # Store rollout
        self.reasoning_rollout_store.append(rollouts)
        
        # Store complete response
        self.reasoning_string_complete_store.append(response_string)
        
        # Process and store clean reasoning
        processed = response_string.replace("<think>", "")
        if "</think>" in processed:
            processed = processed.split("</think>")[0]
        self.curr_reasoning = processed.strip()
        self.reasoning_string_store.append(self.curr_reasoning)
    
    def update_summarization(self, rollouts, response_string):
        # Similar to update_reasoning
        self.summarization_rollout_store.append(rollouts)
        self.summarization_string_complete_store.append(response_string)
        
        # Process and update current summary
        processed = response_string.replace("<think>", "").replace("</think>", "").strip()
        self.curr_summary = processed
        self.summarization_string_store.append(self.curr_summary)
```

### Queue Management

- **Problem Queue**: Stores `InferenceProblemState` objects
- **Result Queue**: Stores lists of `RolloutResult` objects
- **Shared Memory**: Configured via `shared_memory_entry_size`

If you see queue full errors:
- Increase `problem_queue_size` or `result_queue_size`
- Increase `shared_memory_entry_size` for larger states

## Troubleshooting

### Problem: Summaries don't improve
**Solution**: Refine summarization prompt to be more specific

### Problem: Running out of memory
**Solution**: 
- Reduce `num_reasoning_steps`
- Increase `shared_memory_entry_size`
- Clear state history periodically

### Problem: Slow throughput
**Solution**:
- Reduce `num_reasoning_steps`
- Increase `llm_max_rollouts`
- Add more LLMs
- Increase `rollout_workers`

### Problem: Prompts not filling correctly
**Solution**:
- Check template variables: `{problem}`, `{curr_summary}`, `{existing_summary}`, `{reasoning}`
- Verify `InferenceProblemState` has correct attributes
- Check prompt template format strings

## Advanced Usage

### Early Stopping

Add logic to stop if solution is found:

```python
for step_idx in range(num_reasoning_steps):
    # Do reasoning
    solution_result = await solution_rollout_policy(...)
    
    # Check if solved
    if is_solution_correct(solution_result, problem_state.answer):
        logger.info(f"Solved at step {step_idx + 1}, stopping early")
        break
    
    # Continue with summarization
    summarization_result = await summarization_rollout_policy(...)
```

### Adaptive Steps

Vary steps based on problem difficulty:

```yaml
actor:
  num_reasoning_steps_easy: 2
  num_reasoning_steps_hard: 5
  difficulty_threshold: 0.7
```

### Custom State Management

Extend `InferenceProblemState` for domain-specific needs:

```python
class MathProblemState(InferenceProblemState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.equations_found = []
        self.techniques_used = []
```

## Performance Tips

1. **Batch Problems**: Ensure problem queue stays full for maximum utilization
2. **Tune LLM Parameters**: Adjust `max_tokens` per step vs total problem
3. **Monitor Queues**: Watch `problem_queue_size` and `result_queue_size` in logs
4. **Profile Bottlenecks**: Is it LLM generation, queue I/O, or result processing?
5. **Scale Horizontally**: Add more `rollout_workers` for more parallelism

## Summary

Online RC rollouts provide a powerful way to tackle complex problems through iterative reasoning and summarization. The key insight is maintaining state (via summaries) across multiple cycles, allowing the model to build on its own progress.

**Quick Start:**
1. Set `num_reasoning_steps: 3`
2. Define reasoning and summarization prompts
3. Run and monitor throughput vs standard actor
4. Adjust based on problem complexity and resources
