---
name: research-artifacts
description: |
  Load when planning or logging LLMG research campaigns. Work-dir layout,
  stage gates, and links to the autoresearch harness.
disable-model-invocation: true
---

# Research Artifacts

Work items live under `$(meridian work current)/`. Suggested files:

| File | Owner | Purpose |
|---|---|---|
| `charter.md` | research-lead | Link + one-paragraph thesis (source: blog proposal) |
| `hypothesis.md` | research-lead | Falsifiable claim for this campaign |
| `stage.md` | research-lead | Which roadmap stage (0–4); entry/exit criteria |
| `experiment-log.md` | research-lead | Human-readable narrative; mirror key rows from `results.tsv` |
| `decisions/` | research-lead | Stage decisions worth KB promotion |

## Charter source

https://haowjy.github.io/blog/layered-latent-memory-grafts/

## Harness (repo root — not in work dir)

| File | Role |
|---|---|
| `program.md` | Instructions for **experiment-runner** autonomous loop |
| `train.py` | Only file the runner edits |
| `prepare.py` | Fixed eval/data — read only |
| `results.tsv` | Tab-separated experiment log (untracked) |

## Stage gates

Before spawning **experiment-runner**, confirm in `stage.md`:

- GPU/data ready (`uv run prepare.py` done)
- Metric and budget named (val_bpb, 5-minute wall clock)
- Branch tag agreed (`autoresearch/<tag>`)

After a campaign: promote durable findings to KB (`decisions/`, wiki pages per
`/kb-conventions`); run `meridian work done` when the stage is closed.
