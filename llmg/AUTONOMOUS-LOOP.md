# Autonomous LLMG research loop

This file is the operating contract for a Codex `/goal` loop in this repo.
The loop runs on `main` and assumes exactly one local GPU is available.

## Objective

Improve the LLMG research program by running repeated short experiment cycles.
Each cycle should test one concrete hypothesis, inspect evidence, decide whether
to keep, revise, or abandon it, and record the result.

The loop should keep moving until the user pauses/stops it, the current path is
genuinely blocked, or no defensible next experiment remains under the current
hardware and data constraints.

## Operating mode

- Work directly on `main`; do not create an experiment branch unless the user
  changes this policy.
- Treat the GPU as an exclusive resource. Never run two GPU experiments at the
  same time.
- CPU-only readers, explorers, and analysis jobs may run while the GPU is busy
  only if they cannot interfere with the active run's files.
- Use calibrate/smoke runs for search. Use official runs only to promote a
  result that already looks better under calibrate evidence.
- Keep edits small and hypothesis-scoped. Avoid broad refactors unless needed
  to run or measure the experiment.
- Preserve unrelated dirty work. Do not revert files unless the loop itself
  changed them and the attempt is being discarded.

## Cycle shape

1. Review current status: `RESEARCH-LOG.md`, `llmg/ROADMAP.md`, recent
   `llmg/runs/*/summary.md`, and the active experiment README/config.
2. Choose one hypothesis from the strategy ladder or add a new justified one.
3. Make the smallest code/config/data change needed to test it.
4. Run a short calibrate/smoke command with an approximate 5-minute target
   budget.
5. Inspect metrics, run logs, train/validation signals when present, GPU memory,
   trace samples, and failure cases.
6. Decide:
   - keep if it improves the target metric without breaking guards,
   - revise if the failure is informative and a small fix is obvious,
   - abandon/revert loop-owned changes if the hypothesis is worse or unstable.
7. Record the attempt in the loop log.
8. Reflect on the next highest-leverage hypothesis before continuing.

## Metrics

Default starting target is `P1-02`, because Phase 0 retrieval is saturated and
answer quality is weak.

Primary metrics:

- `answer_em`
- `answer_cosine_hit_rate`
- answered-only cosine / answered-only hit rate when available

Guard metrics:

- `retrieval_recall@5` should not materially regress versus the relevant
  Phase 0 or previous P1 baseline.
- `temporal_recall@5` should not materially regress without a documented reason.
- Runs must stay inside local GPU memory after reasonable mitigations.
- Calibrate/smoke runs should remain close to the intended short-run budget.

If the active strategy is P0-TW-05 or another harder-eval path, the primary
metric can switch to `retrieval_recall@5` separation on the hard subset.
Record the switch in the loop log.

## First commands

Prefer short probes before full runs:

```bash
uv run python -m llmg.run --list
uv run python -m llmg.run --experiment P1-02 --run-phase calibrate \
  --param calibrate_max_train_steps=10 \
  --param calibrate_max_train_rows=64 \
  --param calibrate_max_eval_rows=5
```

If CUDA fragmentation appears likely, run GPU experiments with:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m llmg.run --experiment P1-02 --run-phase calibrate \
  --param calibrate_max_train_steps=10 \
  --param calibrate_max_train_rows=64 \
  --param calibrate_max_eval_rows=5
```

## OOM policy

On OOM, inspect the traceback and memory report, then try the smallest relevant
mitigation in this order:

1. Lower `max_eval_rows` / `calibrate_max_eval_rows`.
2. Lower `max_train_rows`.
3. Lower `max_train_steps`.
4. Lower `max_seq_len`.
5. Lower `max_article_chars`.
6. Lower `lora_rank`.
7. Keep `train_batch_size=1` and use gradient accumulation instead of larger
   batches.
8. Confirm train and eval still run in separate subprocesses.
9. Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

If the same hypothesis OOMs after three distinct mitigations and no new
evidence points to a fix, abandon that hypothesis and record why.

## Strategy ladder

Start near the highest evidence-to-cost ratio:

1. Debug P1-02 train/eval adapter behavior and calibrate-vs-official mismatch.
2. Improve answer-quality analysis: answered-only metrics, no-submit counts,
   trace sampling, and failure buckets.
3. Try cleaner SFT data variants: answer-only, shorter article context,
   relation/object formatting, and reduced distractor text.
4. Try protocol-learning data: retrieved-context examples, tool-trace
   distillation, or explicit "retrieve then answer" chat traces.
5. Implement/run P0-TW-05 BM25-hard subset if retrieval saturation blocks
   progress on current evals.
6. Try fast/slow LoRA or rank-matched controls only after naive/protocol LoRA
   baselines are stable.
7. Try temporal-gated LoRA/ReFT-style scaffolds after there is a hard eval that
   can expose temporal behavior.
8. Import or build additional datasets only when local TemporalWiki variants no
   longer answer the current research question.

The loop may add new strategies when evidence justifies them. Record the reason
before spending GPU time.

## Logging

Maintain a compact log for every attempt. Prefer the active Meridian campaign
log when available; otherwise use a local untracked loop log under `llmg/runs/`.

Minimum row/entry fields:

- timestamp
- attempt id
- hypothesis
- changed files
- command
- run directory
- primary metrics
- guard metrics
- peak memory / OOM notes
- status: `keep`, `revise`, `abandon`, `blocked`, or `promote`
- interpretation
- next hypothesis

Do not paste full TSVs into `RESEARCH-LOG.md`. Update `RESEARCH-LOG.md` only
for new official bests, gate pass/fail, surprising official results, or campaign
close, following `AGENTS.md`.

## Promotion rule

Promote a calibrate result to official only when:

- the improvement is tied to a named hypothesis,
- the metric moved in the intended direction,
- guard metrics did not show an obvious regression,
- the run completed without unresolved OOM or trace corruption,
- the change is small enough to understand or has been documented.

Official promotion should update the appropriate research logs if it changes
the headline program state.

## Stop or pause conditions

Pause and report instead of spinning when:

- the next step requires a dataset/model download and network or credentials are
  unavailable,
- all plausible OOM mitigations for the current hypothesis have failed,
- the active metric is invalid or the harness is producing inconsistent
  artifacts,
- continuing would require destructive git operations on changes not owned by
  the loop,
- the user interrupts, pauses, or redirects the goal.
