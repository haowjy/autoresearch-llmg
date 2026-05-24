# P0-TW-01b — retention (BM25)

> **Status: deprecated** — exploratory Gate 0 history only (collapsed index). **Canonical baselines:** [P0-TW-03](../P0-TW-03/) harness BM25 `train_stable` + `stable` on corpus v2 — **~0.78** subject / **~0.78** temporal recall@5; see [phase0-baselines][phase0-baselines]. Supersedes this experiment’s retention probe.

**Gate:** none (companion to deprecated [P0-TW-01](../P0-TW-01))

| Field | Value |
|-------|--------|
| Run mode | `eval_only` |
| Data | [saxenan3/temporalwiki-drift-cl-easy][tw-easy] |
| Index | `train` + `stable` articles |
| Eval | `stable` split (retention probe) |
| Primary metric | `retrieval_recall@5` |

Also reports `retrieval_recall@5_test_same_index` — acquisition recall on **test** using the same train+stable index (sanity vs P0-TW-01 train-only index).

```bash
uv run python -m llmg.run --experiment P0-TW-01b
```

---

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[phase0-baselines]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/phase0-baselines.md
