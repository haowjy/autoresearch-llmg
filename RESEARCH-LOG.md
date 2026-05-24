# LLMG research log (program index)

**Central index** for the full experiment program. Curated in this repo (git-tracked).

| Layer | Path | Role |
|-------|------|------|
| **This file** | `RESEARCH-LOG.md` | Headline stats + pointers to campaigns and machine logs |
| **Campaign narrative** | `llmg/work/<slug>/experiment-log.md` in `haowjy/research-docs` | Interpretation, surprises, stage context |
| **Machine runs** | `results.tsv` + `llmg/runs/<timestamp>_<id>/` | Every `llmg.run` attempt (local, untracked runs) |

Charter: [Layered Latent Memory Grafts][charter-blog] · Setup: [DEVELOPMENT.md][development]

---

## Headline stats (official runs)

Best **official** score per `experiment_id` on this machine (from [results.tsv][results-tsv]).  
Gate decisions use `run_phase=official` only.

| Experiment | Status | Primary metric | Best score | Commit | Latest run |
|------------|--------|----------------|------------|--------|------------|
| P0-TW-03 | done | `retrieval_recall@5` (hybrid agent, test; Wave B) | **1.0000** | `15e6c19` | `llmg/runs/20260524-023355_P0-TW-03` |
| P1-02 | planned | `retrieval_recall@5` (LoRA + RAG) | — | `15e6c19` | — |

*Last updated: 2026-05-24. **Canonical Phase 0 RAG** = [P0-TW-03][p0-tw-03] corpus v2 (~1004 versioned `doc_id`s). Harness BM25 `test` subject recall@5 **~0.91** (supersedes collapsed-index **0.93**).*

### Deprecated (collapsed index — archaeology)

Superseded by [P0-TW-03][p0-tw-03] v2 harness BM25 cells; runners remain for history.

| Experiment | Status | Primary metric | Best score | Commit | Latest run |
|------------|--------|----------------|------------|--------|------------|
| ~~P0-TW-01~~ | deprecated | `retrieval_recall@5` (test, train index) | 0.9333 | `657edff` | `llmg/runs/20260523-123136_P0-TW-01` |
| ~~P0-TW-01b~~ | deprecated | `retrieval_recall@5` (stable, train+stable index) | 0.7600 | `70d57d6` | `llmg/runs/20260523-154507_P0-TW-01b` |

### P0-TW-03 pinned cells (corpus v2)

Orchestrator: [P0-TW-03][p0-tw-03]. Calibrate: [run-p0-tw-03-cal-v2][run-p0-tw-03-cal-v2]. Official: [run-p0-tw-03-off-v2][run-p0-tw-03-off-v2]. KB: [phase0-baselines].

| Cell | recall@5 | temporal@5 | answer_em | answer_cosine | notes |
|------|----------|--------------|-----------|---------------|-------|
| fs + `agent_term_basic` + `test` | **0.8400** | 0.6467 | 0.2867 | 0.426 | shell/rg; ~2.9 cmds/row |
| fs + `agent_term_hybrid` + `test` | **1.0000** | 0.9733 | 0.3400 | 0.475 | `search_hybrid`; 0 cmds |
| fs + `agent_term_basic` + `stable` | **0.9400** | 0.9400 | 0.4200 | 0.740 | vs deprecated 01b collapsed BM25 **0.76** |
| fs + `agent_term_hybrid` + `stable` | **0.9600** | 0.9600 | 0.4200 | 0.773 | hybrid tool |
| sqlite + `agent_term_basic` + `stable` | **0.9400** | 0.9400 | 0.4200 | 0.740 | same as fs shell on stable |
| fs + harness_bm25 + `test` (parity) | **0.9067** | 0.8600 | — | — | harness on exported corpus |

Official Wave B ~84 min. Traces: structured JSONL under `agent_traces/`. Prior pre-v2 runs: [run-p0-tw-03-off][run-p0-tw-03-off].

Calibrate Wave A ([run-p0-tw-03-cal-v2][run-p0-tw-03-cal-v2]): **BM25** recall/temporal **0.91** / **0.86** (`test`), **0.78** / **0.78** (`stable`); **hybrid** **0.97** / **0.95** (`test`), **0.90** / **0.90** (`stable`); **rg** ~0.02 / 0.01 (install [ripgrep][ripgrep] for real rg baseline).

Shell agent (`test`): **0.84** recall / **0.65** temporal — pinned as `retrieval_recall@5_shell` in run metrics.

### Answer quality (Gemma 4, `submit_answer`)

Headline `answer_cosine` (~0.43–0.48) blends **empty submits** (no `submit_answer` before step limit). Re-score from traces ([analyze-agent-answers][analyze-agent-answers]):

| Cell | EM (all rows) | cos≥0.85 (all) | cos mean (answered only) | cos≥0.85 (answered) | no submit |
|------|---------------|----------------|--------------------------|---------------------|-----------|
| shell, `test` | 0.29 | 0.29 | **0.65** | **0.43** | 51/150 |
| hybrid, `test` | 0.34 | 0.34 | **0.65** | **0.47** | 41/150 |

Among **answered** rows, ~**43–47%** are semantically on-target (MiniLM ≥0.85); ~**61%** have cosine ≥0.5. EM and cosine-hit rates match because short factual answers either match gold `object` or fall below 0.85. Retrieval can succeed while answers stay weak: shell **49%** subject-hit but cosine below 0.5; **29** rows have subject recall but wrong temporal slice.

### Program snapshot

- **Phase 0 (canonical):** [P0-TW-03][p0-tw-03] on [TemporalWiki][tw-easy] corpus v2 — harness BM25 **0.91** / **0.86** (`test` subject / temporal recall@5), **0.78** / **0.78** (`stable`); hybrid harness **0.97** / **0.90**; agent shell **0.84** / **0.94** (temporal **0.65** / **0.94**); hybrid agent **1.0** / **0.96** retrieval; `answer_em` **0.29–0.42** with `submit_answer` validation.
- **Gate 0:** passed on v2 matrix (2026-05-24); see [stage.md][campaign-stage].
- **Deprecated (archaeology):** [P0-TW-01][p0-tw-01] / [P0-TW-01b][p0-tw-01b] — collapsed-index BM25 (**0.93** / **0.76** official); not comparable to v2 without re-baseline.
- **Next:** **P1-02** QLoRA calibrate; reduce no-submit rate; optional `ripgrep` for rg harness row.

See also: [EXPERIMENTS.md][experiments] · [ROADMAP.md][roadmap] · [DATASETS.md][datasets]

---

## Datasets (catalog)

**In harness:** [TemporalWiki drift (easy)][tw-easy] — canonical **P0-TW-03**; deprecated **P0-TW-01** / **P0-TW-01b** (archaeology).

**Roadmap / literature:** [full dataset index][datasets] (StreamingQA, PAT-Questions, ChronoQA, TemporalWiki 2022, narrative benchmarks, unlearning refs, custom repo/story plans).

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
uv run python -m llmg.run --experiment P0-TW-03 --run-phase official
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
[development]: DEVELOPMENT.md
[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[experiments]: llmg/EXPERIMENTS.md
[roadmap]: llmg/ROADMAP.md
[datasets]: llmg/DATASETS.md
[p0-tw-01]: llmg/experiments/P0-TW-01
[p0-tw-01b]: llmg/experiments/P0-TW-01b
[p0-tw-03]: llmg/experiments/P0-TW-03
[run-p0-tw-03-cal]: llmg/runs/20260523-180436_P0-TW-03
[run-p0-tw-03-cal-v2]: llmg/runs/20260524-020649_P0-TW-03
[run-p0-tw-03-off]: llmg/runs/20260523-231910_P0-TW-03
[run-p0-tw-03-off-v2]: llmg/runs/20260524-023355_P0-TW-03
[run-p0-tw-03-hybrid]: llmg/runs/20260524-001722_P0-TW-03
[analyze-agent-answers]: scripts/analyze_agent_answers.py
[ripgrep]: https://github.com/BurntSushi/ripgrep#installation
[run-p0-tw-03-off-legacy]: llmg/runs/20260523-180602_P0-TW-03
[phase0-baselines]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/phase0-baselines.md
[run-p0-tw-01b]: llmg/runs/20260523-154507_P0-TW-01b
[run-latest]: llmg/runs/20260523-123136_P0-TW-01
[runs-latest]: llmg/runs/latest
[results-tsv]: results.tsv
[runs-readme]: llmg/runs/README.md
[campaign-log]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/experiment-log.md
[campaign-stage]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/stage.md
[kb-index]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/index.md
[wiki-loop]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/autoresearch-loop.md
[wiki-registry]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/experiment-registry.md
