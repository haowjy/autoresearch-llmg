# llmg — post-training hacking harness

Autoresearch loop over **frozen base + LoRA / RAG eval**, not pretrain.

**Setup:** [DEVELOPMENT.md][development] · **Program log:** [RESEARCH-LOG.md][research-log]

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

Charter: [Layered Latent Memory Grafts][charter-blog]

KB registry: `llmg/kb/wiki/experiment-registry.md` in [research-docs][research-docs-repo]

---

## References

[development]: ../DEVELOPMENT.md
[research-log]: ../RESEARCH-LOG.md
[charter-blog]: https://haowjy.github.io/blog/layered-latent-memory-grafts/
[research-docs-repo]: https://github.com/haowjy/research-docs
