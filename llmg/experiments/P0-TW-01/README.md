# P0-TW-01

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
