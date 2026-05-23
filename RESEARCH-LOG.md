# LLMG research log (program index)

**Central index** for the full experiment program. Curated in this repo (git-tracked).

| Layer | Path | Role |
|-------|------|------|
| **This file** | `RESEARCH-LOG.md` | Headline stats + pointers to campaigns and machine logs |
| **Campaign narrative** | `llmg/work/<slug>/experiment-log.md` in `haowjy/research-docs` | Interpretation, surprises, stage context |
| **Machine runs** | `results.tsv` + `llmg/runs/<timestamp>_<id>/` | Every `llmg.run` attempt (local, untracked runs) |

Charter: [Layered Latent Memory Grafts][charter-blog]

---

## Headline stats (official runs)

Best **official** score per `experiment_id` on this machine (from [results.tsv][results-tsv]).  
Gate decisions use `run_phase=official` only.

| Experiment | Status | Primary metric | Best score | Commit | Latest run |
|------------|--------|----------------|------------|--------|------------|
| P0-TW-01 | done | `retrieval_recall@5` | **0.9333** | `657edff` | `llmg/runs/20260523-123136_P0-TW-01` |
| P0-TW-03 | planned | hybrid recall@5 | — | — | — |
| P1-02 | planned | TBD (LoRA + RAG) | — | — | — |

*Last updated: 2026-05-23. Agents: refresh this table after each new official experiment or best-score change.*

### Program snapshot

- **Phase 0 (RAG floor):** [P0-TW-01][p0-tw-01] passed — BM25 on [TemporalWiki][tw-easy] `train` index, **93.3%** recall@5 on `test` (150 rows, no LLM).
- **Open issue:** `stable` retention eval needs index including `stable` articles (train-only index → 0% recall).
- **Next:** P0-TW-03 hybrid RAG → Gemma answer eval → QLoRA calibrate.

See also: [EXPERIMENTS.md][experiments] · [ROADMAP.md][roadmap]

---

## Campaign logs (narrative)

| Campaign slug | Work path (research-docs) | Stage | Experiment log |
|---------------|---------------------------|-------|----------------|
| `llmg-v1-first-experiment` | `llmg/work/llmg-v1-first-experiment` | Phase 0 — TemporalWiki RAG | `llmg/work/llmg-v1-first-experiment/experiment-log.md` |
| `llmg-bootstrap` | `llmg/work/llmg-bootstrap` | setup | — |

Active campaign: [llmg-v1-first-experiment log][campaign-log] · [stage.md][campaign-stage]

---

## Machine results (all runs)

| Artifact | Path | Notes |
|----------|------|--------|
| Run index | `results.tsv` | Untracked; one row per `llmg.run` |
| Latest run | `llmg/runs/latest` | Symlink to newest timestamp dir |
| Per-run bundle | `llmg/runs/<timestamp>_<experiment_id>/` | `run.log`, `metrics.json`, `summary.md`, `experiment_config.json` |

Details: [runs/README.md][runs-readme]

```bash
uv run python -m llmg.run --experiment P0-TW-01
cat llmg/runs/latest/summary.md
tail -5 results.tsv
```

---

## How to update (agents)

1. **After `llmg.run`** — row in `results.tsv`; artifacts under `llmg/runs/`.
2. **After a meaningful official result** — append campaign `experiment-log.md` (dated section).
3. **Same session** — update **this file**: headline table, snapshot, campaign row; add new `[ref-id]:` lines under [References](#references).
4. **When a decision sticks** — promote to `llmg/kb/decisions/` in research-docs.

Do **not** duplicate every calibrate run here; use the campaign log for narrative.

**Link style:** tables = backtick paths; narrative = `[label][ref-id]`; URLs only in [References](#references).

---

## References

[charter-blog]: https://haowjy.github.io/blog/layered-latent-memory-grafts/
[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[experiments]: llmg/EXPERIMENTS.md
[roadmap]: llmg/ROADMAP.md
[p0-tw-01]: llmg/experiments/P0-TW-01
[run-latest]: llmg/runs/20260523-123136_P0-TW-01
[runs-latest]: llmg/runs/latest
[results-tsv]: results.tsv
[runs-readme]: llmg/runs/README.md
[campaign-log]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/experiment-log.md
[campaign-stage]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/stage.md
[kb-index]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/index.md
[wiki-loop]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/autoresearch-loop.md
[wiki-registry]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/experiment-registry.md
