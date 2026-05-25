# P0-TW-05 — BM25-hard subset on TemporalWiki drift CL (base)

**Status:** spike / not implemented — see campaign [experiment-log][exp-log] § Next (2026-05-25).

## Problem

[P0-TW-04](../P0-TW-04/) on [tw-cl][tw-cl] drops harness BM25 `test` recall@5 to **~0.77** but **hybrid agent** recall stays **1.0** — same saturation as easy [P0-TW-03](../P0-TW-03/). We need an eval where structured `search_hybrid` is not a near-ceiling.

## Proposed design

| Field | Value |
|-------|--------|
| Dataset | Same as P0-TW-04 (`load_tw_cl`, `cl_query_from_row`) |
| Filter | `test` rows where memory harness BM25@5 = 0 on first pass (~**35** / 150 rows at k=5) |
| Matrix | Reuse `tw_matrix_run` with `eval_row_indices` or `eval_filter: bm25_miss` |
| Waves | Official: Wave B agent + harness parity on **filtered** rows only |

## Expected outcome

Hybrid agent recall@5 **< 1.0** on the hard subset; separates tool ceiling from full-split saturation.

## Effort (estimate)

| Task | Time |
|------|------|
| `eval_filter` in `tw_matrix_run` + P0-TW-05 `config.yaml` / thin `runner.py` | ~1–2 h |
| Calibrate smoke (filter sanity) | ~5 min CPU |
| Official Wave B (≈35 rows) | ~20–40 min GPU |

**Blockers:** none beyond implementing row filter + precompute BM25 misses once per run dir.

## Alternatives (deferred)

- **TemporalWiki EMNLP 2022** — new HF schema + corpus export (~4–8 h scaffold).
- **Hybrid tool ablation** (BM25-only tool, k=1) — measures tool design, not dataset difficulty.
- **StreamingQA / PAT** — different domain; Phase 1+.

## Command (when implemented)

```bash
uv run python -m llmg.run --experiment P0-TW-05 --run-phase calibrate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python -m llmg.run --experiment P0-TW-05 --run-phase official
```

## References

[tw-cl]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl
[exp-log]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/experiment-log.md
