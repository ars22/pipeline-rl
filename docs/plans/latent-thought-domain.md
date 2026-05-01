# Plan: Latent-Thought RL Domain

## Context

We want to investigate whether a model can learn to emit useful "latent thought" auxiliary text that, when prepended to a true suffix, makes the joint continuation `(aux, suffix)` *easier on average* for a frozen base model to predict from a `prefix`. This is a research probe — we are not hand-designing the format of `aux`; we want RL to discover it.

Concretely, given `(prefix, suffix)` from a pretraining-style text dataset:

- **Policy** generates `aux ~ p(. | prefix, suffix, prompt)`.
- **Frozen evaluator** (the base LM, no finetuning) scores two quantities:
  - `avg_NLL(suffix | prefix) = -sum_logprob(suffix | prefix) / |suffix|`           (baseline)
  - `avg_NLL(aux, suffix | prefix) = -sum_logprob(aux⊕suffix | prefix) / (|aux|+|suffix|)` (treatment)
- **Reward**: `reward = avg_NLL(suffix | prefix) - avg_NLL(aux, suffix | prefix)`.

The aux string is part of the **target continuation**, not added to the prefix. Positive reward means the policy found auxiliary text that lowers per-token NLL of the overall continuation under the frozen model.

We want to ship this as a new pipelinerl **domain** + a config, reusing existing infrastructure as much as possible.

## Key codebase facts informing the design

1. Domains live under `pipelinerl/domains/<name>/` and follow the 4-file pattern (`__init__.py`, `load_datasets.py`, `rollouts.py`, optional `verifier_api.py`).
2. The rollout entry point is configured via `cfg.actor.rollout_policy = "pipelinerl.domains.<name>.<fn>"` and is called as `await fn(cfg, llm, problem, session)` from `actor.py:schedule_rollouts`.
3. `TrainableLLM.get_batch_logprobs_token_ids(prompt_token_ids, completion_token_ids)` already exists (used in `preprocess.py:81`) — given prefix token IDs and completion token IDs, it returns per-token logprobs from the vLLM server. This is exactly what we need to compute `avg_NLL`.
4. `pipelinerl/world.py` already places a separate **frozen** vLLM server called `preprocessor_llm` (port 8180+). vllm0.py weight updates only target the policy server, so the preprocessor server is genuinely frozen at the initial model.
5. `pipelinerl/rc_actor.py:1482-1611` shows the working pattern for threading an *extra* LLM URL set through to the actor process: `launch.py` forwards `+me.<urls>=...`, the actor reads `cfg.me.<urls>`, builds `TrainableLLM` instances, calls `wait_for_inference_servers`, and passes them into the rollout loop.
6. The math/aicsi/aicsi_rubric domains do not need a custom `verifier_api.py` for this case — the "verifier" is just a logprob query against the frozen vLLM server.

## Design summary

We add a **new `evaluator_llm` job kind** (parallel to `preprocessor_llm` and `summarization_llm`). It hosts a frozen vLLM server for an arbitrary base model — independent of the policy's init model and independent of the KL/preprocessor infrastructure.

This decoupling is required because the user wants flexibility like "train `qwen3-4b-instruct`, evaluate aux against `qwen3-4b-base`." Reusing `preprocessor_llm` would not work cleanly here: `launch.py:run_ref_llm` hardcodes `model_id = cfg.model_path` (the policy init), and overloading that field would either break policy initialization or silently corrupt KL reference logprobs if KL is ever enabled.

The new `evaluator_llm` mirrors the `summarization_llm` pattern in `world.py` and `launch.py`, which is the established way to add a third inference role to the world.

## Files to create / modify

### New files

- `pipelinerl/domains/latent_thought/__init__.py`
  - Re-exports `load_datasets` and `generate_latent_thought_rollout`.

- `pipelinerl/domains/latent_thought/load_datasets.py`
  - `load_datasets(dataset_names, seed, min_side_chars=256, max_total_chars=24000, text_field="text")`:
    - For each spec in `dataset_names`, call `datasets.load_dataset(spec["hub_id"], name=spec.get("config"), split=spec.get("split","train"))`.
    - For each row, take `row[text_field]`, normalize whitespace minimally, find **paragraph boundary offsets** by scanning for `\n\n` (also accept `\r\n\r\n`). The candidate cut offsets are the byte positions immediately after each `\n\n` block.
    - Filter candidates so that both `len(prefix)` and `len(suffix)` are `>= min_side_chars`, and `len(prefix) + len(suffix) <= max_total_chars` (this is the load-time bound; the rollout will additionally enforce a token-level cap once the evaluator tokenizer is available).
    - Drop rows with zero surviving candidates.
    - Build a problem dict:
      ```python
      {
          "dataset": name,
          "id": i,
          "text": full_text,                # keep full text; cut is chosen at rollout time
          "cut_offsets": [int, int, ...],   # surviving candidate cut positions
      }
      ```
    - Shuffle with `seed`. Return a list of problem dicts (matches the contract used by math/aicsi loaders).
  - Slicing by characters keeps the loader tokenizer-agnostic; final token-length enforcement happens in the rollout where the evaluator tokenizer is in scope.
  - Default to a small public dataset (e.g. `wikitext` / `wikitext-103-raw-v1`) so smoke tests work without auth.

- `pipelinerl/domains/latent_thought/rollouts.py`
  - `class Metrics(BaseMetrics)` — adds `avg_nll_baseline`, `avg_nll_treatment`, `aux_tokens`, `suffix_tokens`, `prefix_tokens`, `nll_delta`, `cut_offset`, **`suffix_overlap_ratio`** (longest common substring of aux and suffix, divided by `min(|aux|, |suffix|)` — logged only, NOT used in reward; see "Known v0 limitation" below).
  - `async def generate_latent_thought_rollout(cfg, llm, problem, session, *, evaluator_llm)`:
    1. **Pick a cut online.** From `problem["cut_offsets"]`, sample one offset uniformly at random. `prefix = problem["text"][:offset]`, `suffix = problem["text"][offset:]`.
    2. **Token-level length enforcement.** Tokenize `prefix` and `suffix` with `evaluator_llm.tokenizer`. If `len(prefix_ids) + len(suffix_ids) > cfg.latent_thought.max_total_tokens` (e.g. 8k/16k/32k), trim the longer side from the far end (truncate `prefix` from the left, `suffix` from the right) until the total fits, and rebuild text from token IDs (`tokenizer.decode`). Also enforce `min_side_tokens` (e.g. 64); if either side falls below, skip the sample by returning an empty `RolloutResult` (the actor loop tolerates zero-output rollouts).
    3. **Build the policy prompt** from `cfg.actor.system_prompt` + `cfg.actor.task_template.format(prefix=prefix_text, suffix=suffix_text)`. The template instructs the model to emit auxiliary text — no required format; the reward shapes it.
    4. `llm_call = await llm_async_generate(llm, prompt, session)` — produces `aux` text.
    5. Strip reasoning delimiters (`</think>`) using the same helper math/aicsi use, in case a thinking model is the policy.
    6. **Compute reward via the evaluator.**
       - `aux_ids = evaluator_llm.tokenizer(aux_text, add_special_tokens=False).input_ids`
       - `baseline = evaluator_llm.get_batch_logprobs_token_ids([prefix_ids], [suffix_ids])[0]`
       - `treatment = evaluator_llm.get_batch_logprobs_token_ids([prefix_ids], [aux_ids + suffix_ids])[0]`
       - Sum the per-token logprobs returned (`result["content"][k]["logprob"]`, per the existing usage at `preprocess.py:88`).
       - `avg_nll_baseline = -sum_lp_baseline / len(suffix_ids)`
       - `avg_nll_treatment = -sum_lp_treatment / (len(aux_ids) + len(suffix_ids))`
       - `reward = avg_nll_baseline - avg_nll_treatment`
       - Cap `len(aux_ids) + len(prefix_ids) + len(suffix_ids)` at evaluator's max model len; if exceeded, truncate `aux_ids` from the right and recompute (or skip).
    7. Optional length discount: `reward *= cfg.actor.discount_factor ** llm_call.output_length_tokens` (mirror aicsi). Default `discount_factor: 1.0` since the average-NLL denominator already disincentivizes runaway aux length.
    8. Compute `suffix_overlap_ratio = lcs_len(aux_text, suffix_text) / max(1, min(len(aux_text), len(suffix_text)))` for monitoring only. Do **not** subtract from reward in v00.00.
    9. Build and return `RolloutResult(training_texts=[llm_call.training_text(reward=reward)], metrics=Metrics(...))`.
  - The signature deviates from the standard `(cfg, llm, problem, session)` because we need the evaluator handle. We thread it in via a `functools.partial` at the actor (see modification below) so the dispatcher's existing call site is preserved.

### Modifications to existing files

- `pipelinerl/world.py`
  - In `__init__` (after the `summ_llm_kwargs` block at lines 109-115): add `eval_llm_kwargs = self.cfg.get("evaluator_vllm_config", self.cfg.vllm_config).vllm_kwargs; eval_tp = ...; eval_pp = ...; self.gpus_per_evaluator_llm = eval_tp * eval_pp`. Zero out if `cfg.world.get("evaluator_fraction", 0) == 0`.
  - In `_split_gpus_by_purpose` (lines 215-325): add `evaluator_fraction` into `fraction_sum`, compute `desired_evaluator_gpu_share`, `gpus_per_evaluator`, `self.llms_per_evaluator`, `self.total_evaluator_llms`, subtract its share from `total_finetune_gpus`.
  - In `_place_inference_jobs` (lines 353-441): add a placement loop for `evaluator_llm` (port 8300+), mirroring the `summarization_llm` loop at lines 399-419 exactly.
  - Add `def get_evaluator_urls(self): return [job.url for job in self.get_all_jobs() if job.kind == "evaluator_llm"]`.

- `pipelinerl/launch.py`
  - Add `run_evaluator_llm(cfg, evaluator_llm_idx, local_idx, gpus, exp_dir)` — a near-copy of `run_ref_llm` (lines 85-135), but reads `model_id = cfg.evaluator.model_path` and serves on `8300 + local_idx`. Use `cfg.get("evaluator_vllm_config", cfg.vllm_config)` for vLLM kwargs.
  - Add `"evaluator_llm"` to `all_job_kinds` (line 678) and to the `launch_jobs` lists at lines 1034-1038 (it should be launched alongside `actor`/`actor_llm`, since the actor consumes it).
  - Add the `elif job.kind == "evaluator_llm":` dispatch branch (around line 708) calling `run_evaluator_llm`.
  - In `run_actor`: after computing `llm_urls`, also compute `evaluator_llm_urls = world_map.get_evaluator_urls()` and, if non-empty, append `+me.evaluator_llm_urls={'+'.join(evaluator_llm_urls)}` to the actor command. Mirror the pattern at lines 377-379 for `summarization_llm_urls` in `run_rc_actor`.

- `pipelinerl/actor.py` (around lines 1165-1188 where `train_llms` is built)
  - Read `evaluator_llm_urls = str(cfg.me.evaluator_llm_urls).split("+")` if present.
  - Build `evaluator_llms = [TrainableLLM(base_url=url, model_name=cfg.evaluator.model_path, tokenizer_name=cfg.evaluator.model_path, parameters=..., collect_logprobs=True) for url in evaluator_llm_urls]`. `cfg.evaluator.model_path` is the frozen evaluator model — independent of `cfg.model_path` (policy init).
  - Call `wait_for_inference_servers(evaluator_llm_urls)`.
  - Where `rollout_policy` is dispatched in `schedule_rollouts`, wrap it with `functools.partial(rollout_policy, evaluator_llm=evaluator_llms[i % len(evaluator_llms)])` (round-robin over the evaluator pool, same as policy LLMs).
  - Gate this entire block behind a check like `if "evaluator_llm" in inspect.signature(rollout_policy).parameters` so we don't break math/aicsi rollouts that don't take this kwarg. This keeps the change surgical.

### New config

- `conf/latent-thought-v00.00.yaml` (defaults inherited from `base.yaml`, structure mirrors `aicsi-rl-v00.00.yaml`):
  ```yaml
  actor:
    rollout_policy: pipelinerl.domains.latent_thought.generate_latent_thought_rollout
    system_prompt: null
    task_template: |
      You will be shown a PREFIX and a SUFFIX from a text document. Your task is to generate auxiliary text AUX such that the combined continuation AUX followed by SUFFIX is easy on average for a frozen base language model to predict, given only the PREFIX. The reward depends on the joint per-token predictability of AUX and SUFFIX together — both must be easy to predict; making only one of them easy is not enough.

      Write anything you want. There is no required format.

      PREFIX:
      {prefix}

      SUFFIX:
      {suffix}

      AUXILIARY TEXT:
    discount_factor: 1.0  # disable length discount initially; reward already penalizes long aux through the avg_NLL denominator

  # The frozen evaluator (independent of policy)
  evaluator:
    model_path: Qwen/Qwen3-4B-Base   # e.g. base model whose pretraining loss we want to lower
    # parameters block reserved for future use; we never generate, only score via get_batch_logprobs_token_ids

  # Optional: separate vllm config for the evaluator (TP/PP, max_model_len, etc.)
  # If omitted, world.py falls back to ${vllm_config}.
  # evaluator_vllm_config: ...

  # Policy init (the trainable model)
  model_path: Qwen/Qwen3-4B-Instruct   # e.g. instruct variant we are training

  dataset_loader: pipelinerl.domains.latent_thought.load_datasets
  dataset_loader_params:
    seed: 42
    min_side_chars: 256
    max_total_chars: 24000   # generous loader-side bound; rollout enforces token-level cap below
    text_field: text

  latent_thought:
    max_total_tokens: 16384   # 8k / 16k / 32k as needed; rollout truncates to fit
    min_side_tokens: 64       # skip the rollout if either side falls below this after truncation
  train_dataset_names:
    - hub_id: wikitext
      config: wikitext-103-raw-v1
      split: train
  test_dataset_names:
    - hub_id: wikitext
      config: wikitext-103-raw-v1
      split: validation

  # 5-node layout: 3 nodes actor (24 GPUs), 1 node finetune (8), 1 node evaluator (8).
  # Mirrors the GPU split style of aicsi-rubric-v00.01.yaml.
  world:
    replicas: 1
    actor_fraction: 24
    evaluator_fraction: 8     # the new role; >0 launches frozen evaluator vLLM(s)
    preprocessor_fraction: 0  # KL ref-logprobs disabled; orthogonal to evaluator
    finetune_fraction: 8
    env_replicas: 0           # this domain has no FastAPI environment server
    actor_group_port: 9000

  # Optional: separate vLLM config for the evaluator (e.g. larger max-model-len for prefix+aux+suffix).
  # If omitted, world.py and launch.py fall back to ${vllm_config}.
  evaluator_vllm_config:
    use_v1: false
    vllm_kwargs:
      dtype: bfloat16
      gpu-memory-utilization: 0.9
      tensor-parallel-size: 1
      pipeline-parallel-size: 1
      max-model-len: 32768
      max-num-seqs: 64
      return-tokens-as-token-ids: ""

  finetune:
    rl:
      kl_coef: 0   # no KL penalty; reward signal is purely the NLL-delta we defined

  llm:
    parameters:
      max_tokens: 256   # cap on aux length
      temperature: 1.0
  ```
  Exact GPU fractions, TP/PP, and model names to be tuned by the user; the structure above is what the new domain requires.

## Known v0 limitation: aux=suffix copy degeneracy

The reward formula admits a trivial exploit: if the policy emits `aux = suffix`, then under the evaluator the second half of the joint continuation (`suffix | prefix, suffix`) becomes near-trivial via in-context copy, so `avg_NLL(aux, suffix | prefix) ≈ B/2` and the reward is `~B/2 > 0` for free. Partial overlaps earn proportional rewards.

**Decision for v00.00**: ship as-is. Log `suffix_overlap_ratio` per rollout as a monitoring metric to detect when this strategy dominates. If training learns to copy, fix in a follow-up by either: (a) multiplying reward by `(1 − suffix_overlap_ratio)`, (b) adding an explicit anti-copy line to the prompt, or (c) both.

## Why no `verifier_api.py`

The aicsi/aicsi_rubric domains use a `verifier_api.py` because their judge is an *external* LLM (OpenAI/NVIDIA/Gemini) that returns text. Our "judge" is a logprob query to a vLLM HTTP endpoint — `TrainableLLM.get_batch_logprobs_token_ids` already encapsulates that call, so a separate FastAPI verifier server adds nothing.

## Critical files to read while implementing

- `pipelinerl/domains/aicsi/rollouts.py` — closest template for the new `rollouts.py`.
- `pipelinerl/domains/aicsi/load_datasets.py` — closest template for the new `load_datasets.py`.
- `pipelinerl/preprocess.py:73-92` — exact API contract for `get_batch_logprobs_token_ids` (response shape: `result[i]["content"][k]["logprob"]`).
- `pipelinerl/rc_actor.py:1482-1611` — reference for the `summarization_llm_urls` pattern we mirror for `evaluator_llm_urls`.
- `pipelinerl/actor.py:1165-1230` — where the actor-side LLM list and `wait_for_inference_servers` calls live.
- `pipelinerl/launch.py:85-135` (`run_ref_llm`) — template for our new `run_evaluator_llm`.
- `pipelinerl/launch.py:391-413` (`run_actor`) — where to add the evaluator URL forwarding.
- `pipelinerl/launch.py:678, 1034-1038` — job-kind registry and launch dispatcher.
- `pipelinerl/world.py:109-115, 215-325, 399-419, 472-477` — `summarization_llm` placement, GPU-fraction plumbing, and URL accessor — the exact template to copy for `evaluator_llm`.

## Verification plan

End-to-end smoke test on a small model (e.g. Qwen3-4B) with 1 actor LLM + 1 preprocessor LLM + 1 finetune GPU:

1. **Static check**: `python -c "from pipelinerl.domains.latent_thought import generate_latent_thought_rollout, load_datasets"` imports without error.

2. **Loader check**: Run the new `load_datasets` with `dataset_names=[{"hub_id":"wikitext","config":"wikitext-103-raw-v1","split":"validation"}]` and verify each problem has non-empty `prefix` and `suffix`, no entries are silently dropped because of the length filter.

3. **Reward unit test** (offline, no RL): Spin up a single vLLM server with the base model, then in a Python REPL:
   ```python
   from tapeagents.llms import TrainableLLM
   llm = TrainableLLM(base_url=..., model_name=..., tokenizer_name=..., collect_logprobs=True)
   prefix_ids = llm.tokenizer("Once upon a time", add_special_tokens=True).input_ids
   suffix_ids = llm.tokenizer(" there was a king", add_special_tokens=False).input_ids
   aux_ids    = llm.tokenizer(" — a fairy tale follows.", add_special_tokens=False).input_ids
   r1 = llm.get_batch_logprobs_token_ids([prefix_ids], [suffix_ids])[0]
   r2 = llm.get_batch_logprobs_token_ids([prefix_ids], [aux_ids + suffix_ids])[0]
   ```
   Confirm both calls return `len(suffix_ids)` and `len(aux_ids+suffix_ids)` logprob entries respectively, and that the math `reward = -mean(r1) + mean(r2)` runs without error and produces a finite float. Check sign sanity: an empty / pathological aux should give reward ≈ 0 or negative.

4. **One-step rollout**: Launch the full pipelinerl world via the standard launch script with `latent-thought-v00.00.yaml`, but `--debug` and `train_subset.end=4`. Verify that:
   - `wait_for_inference_servers` succeeds for both policy and evaluator URLs.
   - The actor logs `Using separate evaluator LLMs: [...]` (or equivalent line we add).
   - At least one rollout completes and a `TrainingText` is written to the data stream with a non-zero, finite `reward`.
   - W&B (or logs) show the new `Metrics` fields populated.

5. **Short training run** (~50 steps): Launch full RL training. Watch:
   - Mean reward trends upward (not monotonic, but mean over a window).
   - `avg_nll_treatment` decreases relative to `avg_nll_baseline`.
   - `aux_tokens` does not blow up to `max_tokens` immediately — the avg-NLL denominator should keep aux length somewhat in check.
   - **`suffix_overlap_ratio`** stays low. If it climbs toward 1.0, the policy has discovered the copy degeneracy described above and v00.01 needs the overlap-penalty fix.

If reward is always near zero or always negative across many steps, the immediate suspects are: (a) tokenization mismatch between policy text output and evaluator tokenizer, (b) `add_special_tokens` inconsistency between the prefix/suffix/aux token-id concatenations, (c) the policy collapsing to empty aux. All three are checkable from the metrics we log.
