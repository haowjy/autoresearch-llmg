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
| P0-TW-03 | ok | `retrieval_recall@5` (hybrid agent, test; Wave B) | **1.0000** (v3 / v4) | `f3c3bbe` / `b1b5b86` | v3: `llmg/runs/20260525-144755_P0-TW-03` · v4: `llmg/runs/20260525-230343_P0-TW-03` |
| P0-TW-04 | ok | `retrieval_recall@5` (hybrid agent, test; Wave B) | **1.0000** | `d894068` | `llmg/runs/20260525-184453_P0-TW-04` |
| P0-TW-05 | ok | `retrieval_recall@5` (hybrid agent, BM25-hard `test` subset) | **1.0000** | `b1b5b86` | `llmg/runs/20260529-175800_P0-TW-05` |
| P1-02 | wired | `retrieval_recall@5` (QLoRA + hybrid agent) | — | (run pending) | — |

*Last updated: 2026-05-29. **Canonical Phase 0 RAG (easy)** = [P0-TW-03][p0-tw-03] corpus v2 (~1004 versioned `doc_id`s). Harness BM25 `test` subject recall@5 **~0.91** (supersedes collapsed-index **0.93**). **Pinned agent baselines:** [v3][run-p0-tw-03-off-v3] (6 cells, primary metric) and [v4][run-p0-tw-03-off-v4] (same 6 + `hybrid_deep` ablation, ~3h 36m). **Harder eval (base CL):** [P0-TW-04][p0-tw-04] — [run-p0-tw-04-off][run-p0-tw-04-off] (~78m): harness BM25 **0.77** / hybrid **0.98** `test`; hybrid agent still **1.0** / **0.99**. **BM25-hard subset:** [P0-TW-05][p0-tw-05] — [official][run-p0-tw-05-off] (~10.8m): subset **35/150**, harness BM25 **0.0**, harness hybrid **0.9143** / **0.8571** temporal, hybrid agent **1.0** / **1.0** temporal, answer EM **0.0**. Prior: [v2 (8 steps)][run-p0-tw-03-off-v2]. See [v3 vs v4](#p0-tw-03-baseline-lineage-v3-and-v4) and [shell methodology](#shell-agent-methodology-do-not-over-tune).*

### P0-TW-03 baseline lineage (v3 and v4)

Both are **official** Wave B runs on the same corpus v2 and the same Gemma policy family. Use them together: **v3** for the canonical six-cell matrix; **v4** when you need the tool-ablation row.

| | **v3** | **v4** |
|---|--------|--------|
| **Run** | [20260525-144755][run-p0-tw-03-off-v3] | [20260525-230343][run-p0-tw-03-off-v4] |
| **Code** | [f3c3bbe][commit-f3c3bbe] | [b1b5b86][commit-b1b5b86] |
| **Wave B cells** | **6** | **7** (+ `agent_term_hybrid_deep`) |
| **Primary metric** | hybrid `test` recall@5 **1.0** | same (unchanged) |

**What v3 established (plain English):** We locked in the “raw agent” harness: **16** turns per question, **no** mid-loop coaching, only a **final** nudge to call `submit_answer`, dropped the custom cross-turn KV cache (each step is a normal `model.generate`), capped tool output at **2000** characters to avoid OOM, and installed **`rg`** on the host. Official lanes: shell (`run_shell` + `read_file`), hybrid (`search_hybrid` + `read_file`), stable/sqlite repeats, and harness BM25 parity. Promoted in [d894068][commit-d894068] (docs + registry).

**What v4 added on top of v3 (plain English):** A **seventh** agent lane, `agent_term_hybrid_deep`, that keeps hybrid search but adds **paginated** `read_file` (`offset` / `limit`) and in-file **`grep_file`**. The existing **`agent_term_hybrid`** tool surface was **not** changed so v3 numbers stay comparable. v4 re-ran the full v3 matrix plus this ablation. On easy TemporalWiki, deep did **not** beat hybrid on retrieval (both **1.0** / **0.95**) and answer EM was slightly **lower** (~**0.18** vs **0.21**); `grep_file` was barely used.

**Commits between v3 and v4 (not all are “v4”):** [f3c3bbe][commit-f3c3bbe] → [d894068][commit-d894068] (promote v3) → [f8d32ed][commit-f8d32ed] (P0-TW-04 on base CL) → [b1b5b86][commit-b1b5b86] (hybrid_deep tools + v4 run). The **code diff** for v4 itself is mostly `llmg/agent/tools.py`, `config.yaml` Wave B cell, and matrix metrics — not a policy change to v3 lanes.

### Agent policy (official v3 / v4 — harness default)

| Param | v2 official | v3 (current `config.yaml`) |
|-------|-------------|----------------------------|
| `max_agent_steps` | **8** | **16** |
| Mid-loop nudges | removed earlier | **none** |
| Final-step nudge | — | at `max_steps - 2`: “Answer with submit_answer…” |
| Cross-turn KV | none | removed (v2-style `model.generate` per step) |
| Tool message cap (shell) | 6000 chars | **2000** (VRAM; not a retrieval tweak) |
| `rg` on host | often missing | **yes** ([install-ripgrep][install-rg]) |
| Heuristic bootstrap / `rg`→`grep` | off | off |

Code: [f3c3bbe][commit-f3c3bbe] (16 steps, no cross-turn cache, tool cap 2000). Smoke (2 rows/cell): [run-p0-tw-03-smoke-v3][run-p0-tw-03-smoke-v3]. Official Wave B: [v3][run-p0-tw-03-off-v3] · [v4][run-p0-tw-03-off-v4].

### Deprecated (collapsed index — archaeology)

Superseded by [P0-TW-03][p0-tw-03] v2 harness BM25 cells; runners remain for history.

| Experiment | Status | Primary metric | Best score | Commit | Latest run |
|------------|--------|----------------|------------|--------|------------|
| ~~P0-TW-01~~ | deprecated | `retrieval_recall@5` (test, train index) | 0.9333 | `657edff` | `llmg/runs/20260523-123136_P0-TW-01` |
| ~~P0-TW-01b~~ | deprecated | `retrieval_recall@5` (stable, train+stable index) | 0.7600 | `70d57d6` | `llmg/runs/20260523-154507_P0-TW-01b` |

### P0-TW-04 harder eval (drift CL base)

Dataset: [saxenan3/temporalwiki-drift-cl][tw-cl] — same `train` / `test` / `stable` splits as easy; rows are **triples + articles** (no NL `question`). Eval queries use template `What is the {relation} of {subject} as of {snapshot}?` ([P0-TW-04][p0-tw-04]).

| Phase | run_dir | BM25 `test` | hybrid `test` | status |
|-------|---------|-------------|---------------|--------|
| calibrate | [run-p0-tw-04-cal][run-p0-tw-04-cal] | **0.7667** / 0.74 temporal | **0.9800** / 0.96 temporal | ok |
| official (pinned A+B) | [run-p0-tw-04-off][run-p0-tw-04-off] | **0.7667** / 0.74 | hybrid agent **1.0** / **0.99** | ok (~4667 s) |

#### Official — pinned cells (`d894068`, [run-p0-tw-04-off][run-p0-tw-04-off])

Agent policy = P0-TW-03 v3 (16 steps, tool cap 2000, `rg`). Eval queries: `cl_query_from_row` template.

| Cell | recall@5 | temporal@5 | answer_em | answer_cosine | notes |
|------|----------|--------------|-----------|---------------|-------|
| fs + `agent_term_basic` + `test` | **0.7000** | **0.2867** | 0.0067 | 0.257 | shell |
| fs + `agent_term_hybrid` + `test` | **1.0000** | **0.9933** | 0.0200 | 0.306 | hybrid; primary metric |
| memory + `harness_bm25` + `test` | **0.7667** | **0.7400** | — | — | ↓ ~15 pts vs easy **0.91** |
| memory + `harness_hybrid` + `test` | **0.9800** | **0.9600** | — | — | vs easy **0.97** / **0.95** |
| memory + `harness_bm25` + `stable` | **0.7600** | **0.7600** | — | — | vs easy **0.78** |
| memory + `harness_hybrid` + `stable` | **0.8400** | **0.8400** | — | — | vs easy **0.90** |

vs [P0-TW-03 v3][run-p0-tw-03-off-v3]: hybrid agent recall still **1.0** (not harder for primary metric); shell recall **0.70** (↑ vs **0.69**) but temporal **0.29** (↓ vs **0.37**); harness BM25 separates (~**23%** `test` miss rate).

Easy saturation motivated this hop; base CL separates **harness** lanes but not hybrid-agent recall@5. [P0-TW-05][p0-tw-05] then filtered to BM25-miss rows for a harder retrieval subset.

### P0-TW-05 BM25-hard subset (drift CL base)

P0-TW-05 filters [P0-TW-04][p0-tw-04] `test` rows to the **35/150** examples where train-index harness BM25 misses the gold subject at k=5. It is a retrieval stress subset, not an answer-quality fix.

| Phase | run_dir | BM25 hard | hybrid hard | hybrid agent hard | status |
|-------|---------|-----------|-------------|-------------------|--------|
| calibrate | [run-p0-tw-05-cal-35][run-p0-tw-05-cal-35] | **0.0000** / 0.0000 temporal | **0.9143** / 0.8571 temporal | — | ok |
| official | [run-p0-tw-05-off][run-p0-tw-05-off] | **0.0000** / 0.0000 temporal | **0.9143** / 0.8571 temporal | **1.0000** / 1.0000 temporal | ok (~648 s) |

Interpretation: the hard subset separates BM25 from harness hybrid, but **does not break the structured hybrid-agent retrieval ceiling**. The agent still retrieves the correct subject and temporal slice on every filtered row while answer quality remains weak (`answer_em=0.0`, `answer_cosine_hit_rate=0.0286`).

### P0-TW-03 pinned cells (corpus v2)

Orchestrator: [P0-TW-03][p0-tw-03]. Calibrate: [run-p0-tw-03-cal-v2][run-p0-tw-03-cal-v2]. KB: [phase0-baselines].

#### Official v3 — 16 agent steps (`f3c3bbe`, [run-p0-tw-03-off-v3][run-p0-tw-03-off-v3])

**Canonical six-cell baseline** (2026-05-25, ~8128 s). Policy: 16 steps, no mid-loop nudges, final-step `submit_answer` nudge, cross-turn KV removed, tool stdout cap **2000**, `rg` installed.

| Cell | recall@5 | temporal@5 | answer_em | answer_cosine | notes |
|------|----------|--------------|-----------|---------------|-------|
| fs + `agent_term_basic` + `test` | **0.6933** | **0.3733** | 0.1533 | 0.2785 | shell; ↓ vs v2 (0.84 / 0.65) |
| fs + `agent_term_hybrid` + `test` | **1.0000** | **0.9533** | 0.2000 | 0.3167 | hybrid; **primary metric** |
| fs + `agent_term_basic` + `stable` | **0.9000** | **0.9000** | 0.4000 | 0.7101 | shell stable |
| fs + `agent_term_hybrid` + `stable` | **0.9600** | **0.9600** | 0.3600 | 0.6937 | hybrid stable |
| sqlite + `agent_term_basic` + `stable` | **0.9000** | **0.9000** | 0.4000 | 0.7129 | same as fs shell on stable |
| fs + harness_bm25 + `test` (parity) | **0.9067** | **0.8600** | — | — | harness on exported corpus |

Official Wave B ~2h 15m. Traces: structured JSONL under `agent_traces/`.

#### Official v4 — v3 matrix + `hybrid_deep` (`b1b5b86`, [run-p0-tw-03-off-v4][run-p0-tw-03-off-v4])

**Tool-ablation baseline** (2026-05-25, ~12981 s). Re-runs all v3 cells unchanged in config intent; adds `agent_term_hybrid_deep` (`search_hybrid` + paginated `read_file` + `grep_file`). **Primary metric still pinned to standard hybrid** (not deep).

| Cell | recall@5 | temporal@5 | answer_em | answer_cosine | notes |
|------|----------|--------------|-----------|---------------|-------|
| fs + `agent_term_basic` + `test` | **0.7133** | **0.3133** | 0.1800 | 0.3051 | shell (run variance vs v3) |
| fs + `agent_term_hybrid` + `test` | **1.0000** | **0.9533** | 0.2067 | 0.3355 | hybrid; primary metric |
| fs + `agent_term_hybrid_deep` + `test` | **1.0000** | **0.9533** | 0.1800 | 0.3155 | deep ablation; EM ↓ vs hybrid |
| fs + `agent_term_basic` + `stable` | **0.9000** | **0.9000** | 0.3600 | 0.6655 | shell stable |
| fs + `agent_term_hybrid` + `stable` | **0.9600** | **0.9600** | 0.3400 | 0.6780 | hybrid stable |
| sqlite + `agent_term_basic` + `stable` | **0.9000** | **0.9000** | 0.3600 | 0.6649 | shell stable |
| fs + harness_bm25 + `test` (parity) | **0.9067** | **0.8600** | — | — | bit-identical to v3 |

Official Wave B ~3h 36m. Use **v3** for the canonical headline row; use **v4** to cite the deep-lane negative result.

#### Official v2 — 8 agent steps (`15e6c19`, [run-p0-tw-03-off-v2][run-p0-tw-03-off-v2])

Superseded for **agent** headline rows; kept for comparison. Harness calibrate unchanged.

| Cell | recall@5 | temporal@5 | answer_em | answer_cosine | notes |
|------|----------|--------------|-----------|---------------|-------|
| fs + `agent_term_basic` + `test` | **0.8400** | 0.6467 | 0.2867 | 0.426 | shell/rg; ~2.9 cmds/row |
| fs + `agent_term_hybrid` + `test` | **1.0000** | 0.9733 | 0.3400 | 0.475 | `search_hybrid`; 0 cmds |
| fs + `agent_term_basic` + `stable` | **0.9400** | 0.9400 | 0.4200 | 0.740 | vs deprecated 01b collapsed BM25 **0.76** |
| fs + `agent_term_hybrid` + `stable` | **0.9600** | 0.9600 | 0.4200 | 0.773 | hybrid tool |
| sqlite + `agent_term_basic` + `stable` | **0.9400** | 0.9400 | 0.4200 | 0.740 | same as fs shell on stable |
| fs + harness_bm25 + `test` (parity) | **0.9067** | 0.8600 | — | — | harness on exported corpus |

Official Wave B ~84 min. Traces: structured JSONL under `agent_traces/`.

Prior pre-v2 runs: [run-p0-tw-03-off][run-p0-tw-03-off]. Aborted v3 attempts: [OOM 035045][run-p0-tw-03-off-v3-oom], stale [163430][run-p0-tw-03-off-v3-stale].

Calibrate Wave A ([run-p0-tw-03-cal-v2][run-p0-tw-03-cal-v2]): **BM25** recall/temporal **0.91** / **0.86** (`test`), **0.78** / **0.78** (`stable`); **hybrid** **0.97** / **0.95** (`test`), **0.90** / **0.90** (`stable`); **rg** ~0.02 / 0.01 (install [ripgrep][ripgrep] for real rg baseline).

Shell agent (`test`, v3): **0.69** recall / **0.37** temporal — pinned as `retrieval_recall@5_shell` in run metrics (v2 was **0.84** / **0.65**).

### Shell agent methodology (do not over-tune)

**Lane:** `agent_term_basic` — filesystem store, allowlisted shell (`run_shell`, `read_file`, `submit_answer`). **Not** the Phase 0 retrieval ceiling; that is harness BM25 / **hybrid** agent (`search_hybrid`, recall **1.0** on `test` v3).

**What we measure:** raw Gemma 4 native tool use under a **minimal** system line (“corpus under `articles/`…”). No mid-loop coaching. One final-step `submit_answer` nudge at `max_steps - 2` only.

**Observed behavior (traces, v2 + v3 attempts):**

1. **Ritual search:** `rg -i '<subject>' articles/` → (when `rg` missing) `grep -ri '<subject>' articles/` — every episode; searches the **whole tree**, not one `Subject__slice.md`.
2. **When stuck:** corpus-wide keyword greps from the question, e.g. `grep -ri 'record label' articles/` — matches many unrelated articles; blows context (OOM at 6000-char tool payloads on 24GB GPU).
3. **Temporal gap:** rarely picks the slice where YAML `last_edited` matches question `as_of`; `read_file` returns only the **first 4000** chars of long articles, so facts below the fold look “missing.”
4. **Tool spec nudges (1):** `run_shell` docstring says *Prefer `rg -i 'term' articles/`* — reinforces corpus-wide search. System prompt does **not** mention `rg`.

**Not our fault (model + sparse eval):** generic greps, prose loops, wrong slice, low `submit_answer` rate.

**Our fault (infra only — fix these):**

| Fix | Status |
|-----|--------|
| Install `rg` ([install-ripgrep][install-rg]) | done |
| Tool stdout cap **2000** chars (VRAM probe on stress rows) | default in code |
| Cross-turn KV removed; `model.generate` per step | v3 |

- **2026-05-25 (optional):** `agent_term_hybrid_deep` — paginated `read_file` + in-file `grep_file`; official `agent_term_hybrid` tool surface unchanged for matrix comparability.

**Intentionally not fixing (would confound baselines):** prompt edits for slice/`as_of` discipline, sandbox blocking “bad” greps, raising `read_file` cap, rewriting tool docs. Revisit only as a **forked** ablation (e.g. coached shell) after **P1-02**.

**How to read shell vs hybrid numbers:** shell **0.69 / 0.37** (`test` recall / temporal, v3) is “weak agent + shell tools”; hybrid **1.0 / 0.95** is “same model + structured retrieval tool.” v3 shell `test` dropped vs v2 — likely mix of 16-step context growth, 2000-char tool cap, and same stereotyped grep; not prompt-tuned. Do not optimize shell into hybrid-like behavior and still call it the same baseline.

Campaign narrative: [experiment-log § 2026-05-25][campaign-shell-methodology].

### Answer quality (Gemma 4, `submit_answer`)

Headline `answer_cosine` on v3 `test` is lower than v2 (hybrid **0.32**, shell **0.28**) and still blends **empty submits**. Re-score answered-only from v3 traces ([analyze-agent-answers][analyze-agent-answers]). v2 answered-only reference:

| Cell | EM (all rows) | cos≥0.85 (all) | cos mean (answered only) | cos≥0.85 (answered) | no submit |
|------|---------------|----------------|--------------------------|---------------------|-----------|
| shell, `test` | 0.29 | 0.29 | **0.65** | **0.43** | 51/150 |
| hybrid, `test` | 0.34 | 0.34 | **0.65** | **0.47** | 41/150 |

Among **answered** rows, ~**43–47%** are semantically on-target (MiniLM ≥0.85); ~**61%** have cosine ≥0.5. EM and cosine-hit rates match because short factual answers either match gold `object` or fall below 0.85. Retrieval can succeed while answers stay weak: shell **49%** subject-hit but cosine below 0.5; **29** rows have subject recall but wrong temporal slice.

### Program snapshot

- **Phase 0 (canonical, easy):** [P0-TW-03][p0-tw-03] on [TemporalWiki][tw-easy] corpus v2 — harness BM25 **0.91** / **0.86** (`test`); hybrid harness **0.97** / **0.90**; agent baselines **v3** + **v4** ([v3][run-p0-tw-03-off-v3], [v4][run-p0-tw-03-off-v4]): hybrid **1.0** / **0.95** on `test`; shell ~**0.69–0.71** recall; deep lane does not beat hybrid on answers.
- **Phase 0 (harder, base):** [P0-TW-04][p0-tw-04] on [tw-cl][tw-cl] — harness BM25 **0.77** / hybrid **0.98** `test`; hybrid agent **1.0** / **0.99** ([official][run-p0-tw-04-off]); shell **0.70** / **0.29**.
- **Phase 0 (BM25-hard base):** [P0-TW-05][p0-tw-05] filters [tw-cl][tw-cl] to **35/150** BM25-miss rows — harness BM25 **0.0**, harness hybrid **0.9143** / **0.8571**, hybrid agent **1.0** / **1.0** ([official][run-p0-tw-05-off]); answer EM **0.0**.
- **Gate 0:** passed on v2 matrix (2026-05-24); see [stage.md][campaign-stage].
- **Deprecated (archaeology):** [P0-TW-01][p0-tw-01] / [P0-TW-01b][p0-tw-01b] — collapsed-index BM25 (**0.93** / **0.76** official); not comparable to v2 without re-baseline.
- **Next:** **P1-02** naive QLoRA answer-policy work; P0 retrieval now needs a harder dataset or tool ablation to break hybrid-agent recall.

See also: [EXPERIMENTS.md][experiments] · [ROADMAP.md][roadmap] · [DATASETS.md][datasets]

---

## Datasets (catalog)

**In harness:** [TemporalWiki drift (easy)][tw-easy] — canonical **P0-TW-03**; [drift CL base][tw-cl] — **P0-TW-04** (harder triples); deprecated **P0-TW-01** / **P0-TW-01b** (archaeology).

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
[tw-cl]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl
[p0-tw-04]: llmg/experiments/P0-TW-04
[p0-tw-05]: llmg/experiments/P0-TW-05
[run-p0-tw-04-cal]: llmg/runs/20260525-184358_P0-TW-04
[run-p0-tw-04-off]: llmg/runs/20260525-184453_P0-TW-04
[run-p0-tw-05-cal-35]: llmg/runs/20260529-174749_P0-TW-05
[run-p0-tw-05-off]: llmg/runs/20260529-175800_P0-TW-05
[commit-d894068]: https://github.com/haowjy/autoresearch-llmg/commit/d894068
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
[run-p0-tw-03-off-v3]: llmg/runs/20260525-144755_P0-TW-03
[run-p0-tw-03-off-v4]: llmg/runs/20260525-230343_P0-TW-03
[commit-b1b5b86]: https://github.com/haowjy/autoresearch-llmg/commit/b1b5b86
[commit-f8d32ed]: https://github.com/haowjy/autoresearch-llmg/commit/f8d32ed
[run-p0-tw-03-off-v3-stale]: llmg/runs/20260524-163430_P0-TW-03
[run-p0-tw-03-off-v3-oom]: llmg/runs/20260525-035045_P0-TW-03
[commit-f3c3bbe]: https://github.com/haowjy/autoresearch-llmg/commit/f3c3bbe
[run-p0-tw-03-smoke-v3]: llmg/runs/20260524-144153_P0-TW-03
[run-p0-tw-03-hybrid]: llmg/runs/20260524-001722_P0-TW-03
[analyze-agent-answers]: scripts/analyze_agent_answers.py
[ripgrep]: https://github.com/BurntSushi/ripgrep#installation
[install-rg]: scripts/install-ripgrep.sh
[campaign-shell-methodology]: https://github.com/haowjy/research-docs/blob/main/llmg/work/llmg-v1-first-experiment/experiment-log.md#2026-05-25--shell-agent-methodology-do-not-over-tune
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
