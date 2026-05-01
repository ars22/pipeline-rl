# pipeline-rl

PipelineRL: pipelined RL training for LLMs (CMU-AIRe / HF cluster fork)

## Memory System

This project uses a three-layer memory system:

1. **Global** — `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/` — User preferences and cross-project learnings (shared across ALL projects and clusters)
2. **Cluster** — `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/clusters/mbz/` — Cluster-specific config, hardware notes, and common errors (shared across all projects on this cluster)
3. **Project** — `project-memory/` — Project-specific tracking (this project only)

### On Session Start

1. Read `project-memory/STATUS.md` to understand current project state.
2. Read `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/preferences.md` for user preferences.
3. Read `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/clusters/mbz/cluster_config.md` for cluster environment.
4. If STATUS.md has a "Waiting On" section with job IDs, check their status immediately.
5. If STATUS.md has "Pickup Instructions", follow them — do NOT re-explore the codebase.
6. Do NOT read other project-memory files unless needed for the current task.

### Shared Layer Updates

When you discover something that applies beyond this project, update the appropriate shared layer:

- **Cluster error/fix** → `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/clusters/mbz/cluster_errors.md`
- **Cluster config change** (new partition, module update, etc.) → `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/clusters/mbz/cluster_config.md`
- **Cross-project insight** → `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/cross_project_learnings.md`
- **Library gotcha** → `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/library_notes.md`
- **User preference** → `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/preferences.md`
- **Researcher pattern observed** → `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/research_patterns_raw.md` (append-only — see format in file)

#### Researcher Pattern Observations

Append to `research_patterns_raw.md` when you observe any of the following. Be factual and specific, capturing the *what* and the *why* it's revealing.

**Idea generation & research direction** (highest value):
- User proposes a research direction, hypothesis, or experiment the agent didn't suggest — record the idea AND what reasoning led them there
- User connects results from one project/paper to an idea in another domain
- User reframes a problem in an unexpected way
- User identifies what the *interesting* question is in a set of results

**Judgment & taste**:
- User rejects a result you thought was meaningful, or gets excited about something you'd overlook
- User says "dig deeper" or "that's noise" — what triggered it?
- User insists on a specific control, ablation, or analysis before drawing conclusions

**Methodology & strategy**:
- User chooses a surprising experimental design or prioritization
- User decides to pivot, double down, or abandon a direction — and why
- User specifies what "good enough" evidence looks like for a claim

**Communication**:
- User corrects the agent's approach in a way that reveals a preference

Do NOT update `research_patterns.md` (the summary) — that is only updated when the user explicitly requests it.

### Project Memory

### During Work

As you work, maintain the following files:

- **`project-memory/STATUS.md`** — Update when priorities or blockers change.
- **`project-memory/history.md`** — Append a summary of each session: requests, actions, and **discussion notes** (key ideas explored during open-ended brainstorming, reasoning that led to decisions). Link to any artifacts produced (new entries in functions.md, decisions.md, hypotheses.md). Use date headers.
- **`project-memory/experiments.md`** — When launching or completing experiments, update the tracker.
- **`project-memory/decisions.md`** — When a non-obvious choice is made (approach, library, hyperparameter), log it with the reasoning.
- **`project-memory/functions.md`** — When implementing a reusable function/script, add it here with a one-line description and file path.
- **`project-memory/errors_and_fixes.md`** — When debugging a non-trivial issue, log the error and fix.
- **`project-memory/config.md`** — Keep cluster paths, env vars, and key settings up to date.
- **`project-memory/hypotheses.md`** — When forming or testing a research hypothesis, log it. Link experiments to hypotheses.
- **`project-memory/results_summary.md`** — When meaningful results come in, update the curated summary (not every run — only significant ones).
- **`project-memory/artifact_index.md`** — When creating new output directories, notebooks, or figures, add them here.

### Audit

Run the audit script periodically (at session start or when the user asks) to catch untracked code, stale experiments, and config drift:

```bash
bash /mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/scripts/audit.sh /mnt/weka/home/wen.ye/workspace_m2/workspace/AI-Scientist/repos/pipeline-rl
```

After running, review the warnings and update the relevant project-memory files. Common issues:
- **UNTRACKED CODE**: New .py/.sh files not in functions.md — add them or note they are intentionally excluded.
- **UNTRACKED OUTPUT**: Result directories not in experiments.md or artifact_index.md — register them.
- **STALE**: Experiments marked "running" whose Slurm jobs are no longer active — update their status.
- **DRIFT**: Git branch or conda env doesn't match config.md — update config.md.

### On Session End

When the user says they're done or wrapping up:
1. Update `project-memory/STATUS.md` — this is the most important step. Write it so a fresh agent with ZERO context can immediately continue. Specifically:
   - **Waiting On**: List every running job with its ID, what it does, exact output path, and the exact command to check status.
   - **Pickup Instructions**: Write step-by-step what the next agent should do. Include exact file paths, exact commands, and what to look for. Think: "if I were a new agent reading only STATUS.md, could I continue without searching the codebase?"
   - **Do NOT Redo**: List anything that was tried and failed or is already done.
2. Append session summary to `project-memory/history.md`.
3. Update the cross-project dashboard: `/mnt/weka/home/wen.ye/workspace_m2/workspace/Research-skills/global/active_tasks.md` — update the `## pipeline-rl` section with current status (5-10 lines: what's running, what's done, what's next).

**IMPORTANT**: `/exit` closes the session instantly — the agent gets no chance to act. The user should say "wrapping up" or "done" BEFORE typing `/exit` so the agent can update project-memory. If the session ends abruptly, project-memory may be stale — the next session should verify STATUS.md is current.

### Rules

- Keep STATUS.md under 50 lines. It's a snapshot, not a log.
- In experiments.md, always record: config, command, commit/code state, job ID, status, and key results.
- In experiments.md, link each experiment to a hypothesis in hypotheses.md when applicable.
- In decisions.md, always record the alternatives considered and why they were rejected.
- In results_summary.md, keep it curated — only significant results, not every run.
- Never delete history entries. History, decisions, and hypotheses are append-only.
- Functions.md, experiments.md, and results_summary.md can be updated (e.g., marking an experiment as done).
- When an experiment invalidates a result, mark it as invalid in experiments.md and update results_summary.md.
- Always record Slurm job IDs so we can check job status or retrieve logs later.
