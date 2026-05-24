# P0-TW-01

> **Status: deprecated** — exploratory Gate 0 history only (collapsed subject index). **Canonical baselines:** [P0-TW-03](../P0-TW-03/) harness BM25 on corpus v2 — `test` acquisition **~0.91** subject / **~0.86** temporal recall@5; see [phase0-baselines][phase0-baselines]. Runner kept for archaeology.

**Gate:** none (RAG floor before LoRA)

| Field | Value |
|-------|--------|
| Run mode | `eval_only` |
| Data | [saxenan3/temporalwiki-drift-cl-easy][tw-easy] |
| Method | BM25 on `train` articles, eval `test` |
| Primary metric | `retrieval_recall@5` |

**Official:** edit `config.yaml` only (lock params for gate rows).

**Calibrate:** hack hyperparams via CLI overrides or temporary edits on branch `autoresearch/*`.

```bash
uv run python -m llmg.run --experiment P0-TW-01
uv run python -m llmg.run --experiment P0-TW-01 --param k=10 --param eval_split=stable
```

Campaign log: [experiment-log.md][campaign-log]

---

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[campaign-log]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/experiment-log.md
[phase0-baselines]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/phase0-baselines.md
