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

Root [results.tsv][results-tsv] gets one row per run with a `run_dir` column.

### Agent traces (P0-TW-03+)

Under `agent_traces/<cell_id>/row_<n>.jsonl` — one JSON object per line, typed:

| `type` | Contents |
|--------|----------|
| `episode_start` | question, as_of, gold_subject, gold_answer, model, toolset |
| `assistant_turn` | raw model output, tool_calls, parsed content |
| `tool_result` | tool name + observation (read_file / search_hybrid / submit_answer) |
| `sandbox` | shell cmd, stdout/stderr (truncated), returncode |
| `user_nudge` | loop continuation prompts |
| `episode_end` | final answer, `retrieved_doc_ids`, step/cmd stats |

Use `jq -c 'select(.type==\"tool_result\")' …` to inspect tool observations.

Program index: [RESEARCH-LOG.md][research-log]

These directories are gitignored; only this README is tracked.

---

## References

[results-tsv]: ../../results.tsv
[research-log]: ../../RESEARCH-LOG.md
