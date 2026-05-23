# Experiment IDs

Canonical catalog: [experiment-registry.md](https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/experiment-registry.md)  
(local: `~/.meridian/git/haowjy-research-docs/llmg/kb/wiki/experiment-registry.md`)

Log format in root `results.tsv`:

```
commit	run_phase	experiment_id	primary_metric	score	memory_gb	status	description	run_dir
```

- `run_phase`: `calibrate` | `official` (gates use **official** only)
- `experiment_id`: e.g. `P0-TW-01`, `P1-02`
- `run_dir`: path to per-run artifacts (see below)

**Per-run observability:** each `uv run python -m llmg.run` writes:

```text
llmg/runs/<timestamp>_<experiment_id>/
  run.log  config.json  metrics.json  meta.json  summary.md
```

`llmg/runs/latest` → most recent run. Details: [runs/README.md](runs/README.md).

## Per-experiment layout

Each primary experiment is a directory — **do not edit `llmg/run.py`** to add one:

```text
llmg/experiments/P0-TW-01/
  config.yaml    # official locked params + primary_metric
  runner.py      # run(session=..., **params) implementation
  README.md      # short spec
```

```bash
uv run python -m llmg.run --list
uv run python -m llmg.run --experiment P0-TW-01
uv run python -m llmg.run --experiment P0-TW-01 --param k=10
```

**Calibrate (autoresearch):** edit `llmg/experiment.py` (`ACTIVE_EXPERIMENT`) or hack on branch; runner may also edit `runner.py` on the calibrate branch.

**Official (gates):** only change `config.yaml` in the experiment dir; copy `experiment_config.json` from the run dir as the lock record.
