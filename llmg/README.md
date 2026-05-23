# llmg — post-training hacking harness

Autoresearch loop over **frozen base + LoRA / RAG eval**, not pretrain.

| Path | Role |
|------|------|
| `experiment.py` | **Runner-editable** (calibrate) or reads `experiments/<id>/config.yaml` (official) |
| `run.py` | CLI entry: `uv run python -m llmg.run` |
| `eval/` | Scorecard, RAG baselines |
| `fixtures/` | Tiny corpora |
| `experiments/` | Official per-row configs |
| `runs/` | Timestamped run artifacts |
| `ROADMAP.md` | Phase → campaign → gate |
| `EXPERIMENTS.md` | Link to KB experiment registry |

Charter: https://haowjy.github.io/blog/layered-latent-memory-grafts/

KB registry: `~/.meridian/git/haowjy-research-docs/llmg/kb/wiki/experiment-registry.md`
