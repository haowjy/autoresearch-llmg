---
name: research-artifacts
description: |
  Load when planning or logging LLMG research campaigns. Work-dir layout,
  experiment-log.md narrative, stage gates, and links to the autoresearch harness.
disable-model-invocation: true
---

# Research Artifacts

Work items live under `$(meridian work current)/`. Suggested files:

| File | Owner | Purpose |
|---|---|---|
| `charter.md` | research-lead | Link + one-paragraph thesis (source: blog proposal) |
| `hypothesis.md` | research-lead | Falsifiable claim for this campaign |
| `stage.md` | research-lead | Which roadmap stage (0–4); entry/exit criteria; link to experiment-log |
| `experiment-log.md` | research-lead | **Narrative research log** (see below) |
| `decisions/` | research-lead | Stage decisions worth KB promotion |

## Research log (three layers)

| Layer | Location | Git | Writer |
|---|---|---|---|
| **Program index** | `RESEARCH-LOG.md` at autoresearch-llmg repo root | harness repo | research-lead after official bests / new campaigns |
| **Campaign narrative** | `$(meridian work current)/experiment-log.md` | research-docs | research-lead after meaningful runs |
| **Machine** | `results.tsv` + `llmg/runs/<timestamp>_<experiment_id>/` | runs untracked | `llmg.run` (automatic) |

`RESEARCH-LOG.md` holds **headline stats** (best official per `experiment_id`), links to each campaign `experiment-log.md`, and pointers to `results.tsv` / `llmg/runs/latest/`.

Per-run artifacts (`run.log`, `metrics.json`, `experiment_config.json`, `summary.md`) live under `llmg/runs/`. Campaign logs **interpret** them; the program index **indexes** them.

### When to append

Update **`RESEARCH-LOG.md`** and campaign **`experiment-log.md`** after:

- New **experiment_id** first official (or gate) result
- Result changes **stage.md** status (pass/fail, calibrate done)
- Non-obvious outcomes (metric surprises, split/index bugs, negative results)
- New Meridian work campaign (add row to program index campaign table)
- End of campaign (lessons learned)

**experiment-runner:** do not log every calibrate iteration. research-lead summarizes after reviewing `results.tsv` and `llmg/runs/latest/`.

### Entry contents

Each dated section should include:

1. **Hypothesis** — expected outcome  
2. **Setup** — experiment_id, git commit, params (from `config.yaml` or run dir `experiment_config.json`)  
3. **Result** — primary metric, link to `run_dir`  
4. **Interpretation** — implications for LLMG program / gates  
5. **Surprises / limitations**  
6. **Next** — registry IDs (e.g. P0-TW-03)

Copy the template from the bottom of the active campaign `experiment-log.md` when creating a new work dir.

### Link style

Markdown **reference links**: `[label][ref-id]` in narrative; `[ref-id]: URL` under **`## References`** at bottom. Tables: backtick paths only. Same-repo refs may use relative paths; cross-repo use `https://github.com/haowjy/research-docs/blob/main/...`. Never `file://` or `~/.meridian/...`.

### Promotion

| Stay in log | Promote to KB |
|---|---|
| Run-by-run notes, failed hunches | Accepted decisions → `llmg/kb/decisions/` |
| Provisional next steps | Confirmed gates / vocab → `llmg/kb/wiki/` |

On campaign close: `meridian work done <slug>`; archive promotes work dir per Meridian conventions.

## Charter source

https://haowjy.github.io/blog/layered-latent-memory-grafts/

## Harness (repo — not in work dir)

| File | Role |
|---|---|
| `program.md` | experiment-runner loop instructions |
| `llmg/run.py` | `uv run python -m llmg.run --experiment <ID>` |
| `llmg/experiment.py` | Calibrate entry (runner may edit on branch) |
| `llmg/experiments/<ID>/config.yaml` | Official locked params |
| `llmg/experiments/<ID>/runner.py` | Experiment implementation |
| `results.tsv` | Untracked TSV index (includes `run_dir` column) |
| `legacy/karpathy/` | Archived pretrain harness |

## Stage gates

Before spawning **experiment-runner**, confirm in `stage.md`:

- Data and model weights ready (HF dataset, Gemma license if LoRA)
- Metric and budget named (`primary_metric`, calibrate `wall_seconds` / official `run_budget`)
- Branch tag agreed (`autoresearch/<tag>`)
- `experiment-log.md` exists (create from template if missing)

After meaningful official runs: **append experiment-log.md** before reporting to the human.

After a campaign: promote durable findings to KB; run `meridian work done` when the stage is closed.
