# P0-TW-05 — BM25-hard subset on TemporalWiki drift CL (base)

**Status:** official complete — BM25-hard subset is **35/150** `test` rows; harness hybrid is not saturated, but the hybrid agent still saturates retrieval on the hard subset.

## Problem

[P0-TW-04](../P0-TW-04/) on [tw-cl][tw-cl] drops harness BM25 `test` recall@5 to **~0.77** but **hybrid agent** recall stays **1.0** — same saturation as easy [P0-TW-03](../P0-TW-03/). We need an eval where structured `search_hybrid` is not a near-ceiling.

## Proposed design

| Field | Value |
|-------|--------|
| Dataset | Same as P0-TW-04 (`load_tw_cl`, `cl_query_from_row`) |
| Filter | `test` rows where memory harness BM25@5 = 0 on first pass (~**35** / 150 rows at k=5) |
| Matrix | Thin P0-TW-05 runner writes `hard_subset.json` + `matrix_results.{tsv,json}` |
| Waves | Official: Wave B agent + harness parity on **filtered** rows only |

## Outcome

The subset separates harness BM25 from harness hybrid, but it does **not** break the structured hybrid-agent retrieval ceiling. On the official 35-row run, harness BM25 is **0.0000**, harness hybrid is **0.9143** / **0.8571** temporal, and hybrid agent is **1.0000** / **1.0000** temporal. Answer EM remains **0.0**.

## Effort (estimate)

| Task | Time |
|------|------|
| P0-TW-05 `config.yaml` / thin `runner.py` | done |
| Calibrate smoke (filter sanity) | done |
| Official Wave B (35 rows) | done |

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

Calibrate defaults to the first 10 BM25-miss rows (`max_hard_rows=10`) and skips the agent lane unless `--param run_agent=true` is passed. Use `--param max_hard_rows=35` to evaluate the full hard subset.

## Evidence

| Run | Phase | Rows | BM25 recall@5 | Hybrid recall@5 | Hybrid temporal@5 | Status |
|-----|-------|------|---------------|-----------------|-------------------|--------|
| `20260529-175800_P0-TW-05` | official | full 35 + hybrid agent | **0.0000** | **0.9143** | **0.8571** | hybrid agent recall/temporal **1.0/1.0**, answer EM **0.0** |
| `20260529-175307_P0-TW-05` | calibrate | first 10 + hybrid agent | **0.0000** | **0.8000** | **0.6000** | keep; agent recall/temporal **1.0/1.0**, answer EM **0.0** |
| `20260529-174906_P0-TW-05` | calibrate | first 5 + hybrid agent | **0.0000** | **1.0000** | **0.8000** | keep; agent recall/temporal **1.0/1.0**, answer EM **0.0** |
| `20260529-174749_P0-TW-05` | calibrate | full 35 harness only | **0.0000** | **0.9143** | **0.8571** | keep; confirms non-saturated hybrid retrieval eval |
| `20260529-174649_P0-TW-05` | calibrate | first 10 harness only | **0.0000** | **0.8000** | **0.6000** | keep; filter sanity passed |

## References

[tw-cl]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl
[exp-log]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/experiment-log.md
