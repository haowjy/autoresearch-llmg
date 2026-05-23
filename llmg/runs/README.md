# Run artifacts (per experiment attempt)

Each `uv run python -m llmg.run` creates a timestamped folder:

```text
llmg/runs/20260523-121530_P0-TW-01/
  run.log       # text log (also streamed to stdout)
  config.json   # experiment id, argv, git commit
  metrics.json  # all numeric scores
  meta.json     # elapsed time, hostname, GPU memory
  summary.md    # human-readable recap
```

`llmg/runs/latest` points at the most recent run.

Root [`results.tsv`](../../results.tsv) gets one row per run with a `run_dir` column.

These directories are gitignored; only this README is tracked.
