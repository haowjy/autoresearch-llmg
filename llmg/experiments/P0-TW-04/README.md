# P0-TW-04 — Phase 0 baselines on TemporalWiki drift CL (base)

Harder eval hop: [saxenan3/temporalwiki-drift-cl][tw-cl] (triple + article rows; no NL `question` column). Queries are synthesized as `What is the {relation} of {subject} as of {snapshot}?` — same metrics as [P0-TW-03](../P0-TW-03/).

| | P0-TW-03 | P0-TW-04 |
|---|---|---|
| Dataset | [tw-easy][tw-easy] | [tw-cl][tw-cl] |
| Eval queries | `question` column | `cl_query_from_row` template |
| Splits | `train` / `test` / `stable` | same |

## Commands

```bash
cd ~/gitrepos/research/autoresearch-llmg
hf download saxenan3/temporalwiki-drift-cl --repo-type dataset

# Harness + smoke (CPU)
uv run python -m llmg.run --experiment P0-TW-04 --run-phase calibrate

# Pinned harness + agent cells (GPU for Wave B)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m llmg.run --experiment P0-TW-04 --run-phase official
```

Agent policy matches P0-TW-03 v3: 16 steps, tool cap 2000, `rg` on PATH.

## Official (2026-05-25)

Run [20260525-184453](../../runs/20260525-184453_P0-TW-04) · code `d894068` · ~4667 s · status **ok**.

| Cell | recall@5 | temporal@5 | answer_em | answer_cosine |
|------|----------|------------|-----------|---------------|
| harness BM25 `test` | 0.7667 | 0.74 | — | — |
| harness hybrid `test` | 0.98 | 0.96 | — | — |
| shell agent `test` | 0.70 | 0.2867 | 0.0067 | 0.257 |
| hybrid agent `test` | **1.00** | 0.9933 | 0.02 | 0.306 |
| harness BM25 `stable` | 0.76 | 0.76 | — | — |
| harness hybrid `stable` | 0.84 | 0.84 | — | — |

Hybrid agent recall@5 still **1.0** vs [P0-TW-03 v3](../P0-TW-03/README.md) — harder dataset did not break primary metric. Next: [P0-TW-05](../P0-TW-05/README.md) (BM25-hard subset).

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[tw-cl]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl
