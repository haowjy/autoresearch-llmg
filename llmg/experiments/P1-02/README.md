# P1-02 — protocol LoRA + RAG (Phase 1)

**Status:** scaffold only (training loop not wired).

## Goal

Fine-tune `google/gemma-4-E4B-it` with QLoRA on TemporalWiki [train][tw-easy] Q/A, then evaluate with the same RAG stack as [P0-TW-03][p0-tw-03] (versioned corpus v2, hybrid retrieval optional).

Compare against Phase 0 ceilings:

| Anchor | Metric | P0-TW-03 v2 (harness / agent) |
|--------|--------|-------------------------------|
| Acquisition | `retrieval_recall@5` on `test` | BM25 **0.91**; hybrid agent **1.0** |
| Retention | on `stable` | BM25 **0.78**; hybrid agent **0.96** |
| Answers | `answer_em` / `answer_cosine` | hybrid test EM **0.34**; cosine **0.48** (all rows) |

## Commands (when implemented)

```bash
uv run python -m llmg.run --experiment P1-02 --run-phase calibrate
uv run python -m llmg.run --experiment P1-02 --run-phase official
```

## Implementation checklist

- [ ] QLoRA trainer (4k ctx default; long-context mix per [stage.md][stage])
- [ ] Checkpoint + adapter path in run dir
- [ ] RAG eval hook: reuse `run_harness_cell` / `run_agent_eval` from P0-TW-03
- [ ] Lock `config.yaml` after calibrate wall (~300s proxy)

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[p0-tw-03]: ../P0-TW-03/README.md
[stage]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/stage.md
