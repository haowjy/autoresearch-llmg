# P0-TW-01b — retention (BM25)

**Gate:** none (companion to [P0-TW-01](../P0-TW-01))

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
