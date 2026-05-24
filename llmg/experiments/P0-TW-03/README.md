# P0-TW-03 — phased Phase 0 baseline sweep

Single orchestrator experiment: exports TemporalWiki to **memory / SQLite / filesystem**, runs a **matrix** of harness and agent cells, writes `matrix_results.tsv` under the run dir.

## Waves

| Wave | `--run-phase` | Content |
|------|---------------|---------|
| **A** | `calibrate` | Full-split harness matrix (BM25, hybrid, rg) |
| **C** | `calibrate` | Smoke subset (`wave_c_max_eval_rows: 30`) |
| **B** | `official` | Pinned agent cells (`agent_term_basic` = rg/grep shell, `agent_term_hybrid` = BM25+dense tool) + parity harness |

## Commands

```bash
uv sync
uv run python -m llmg.run --experiment P0-TW-03 --run-phase calibrate
uv run python -m llmg.run --experiment P0-TW-03 --run-phase official
```

Wave B is a **multi-turn agentic loop** on `google/gemma-4-E4B-it`:

1. **System** — search → read → (repeat) → `submit_answer`
2. **Native tools** — `agent_term_basic`: `run_shell`, `read_file`, `submit_answer`; `agent_term_hybrid`: `search_hybrid`, `read_file`, `submit_answer` (see `llmg/search/hybrid.py`)
3. Each turn: full `messages` list is passed through `apply_chat_template` (history stack grows: assistant + tool turns appended); loop continues until `submit_answer` or `max_agent_steps`

Traces: `agent_traces/<cell>/row_<i>.jsonl` — structured episode log (`episode_start` … `assistant_turn` / `tool_result` / `sandbox` … `episode_end`). See [runs/README.md](../../runs/README.md).

```bash
uv run python -m llmg.run --experiment P0-TW-03 --run-phase official
# Smoke agent (5 rows):
uv run python -m llmg.run --experiment P0-TW-03 --run-phase official --param max_eval_rows=5
```

Disable heuristic pre-search (default off): `--param agent_heuristic_bootstrap=false`

**Primary metric (official):** `retrieval_recall@5` is pinned to **hybrid agent + `test`**; shell agent is `retrieval_recall@5_shell` in `metrics.json`.

**Answer analysis:** `uv run python scripts/analyze_agent_answers.py <run_dir> --cell <cell_id>` — splits answered vs no-submit rows.

Skip GPU agent (harness-only):

```bash
uv run python -m llmg.run --experiment P0-TW-03 --run-phase official --param skip_agent=true
```

## Corpus (versioned)

Each **train/stable row** is one document: `articles/<doc_id>.md` with metadata `first_edited`, `last_edited`, `slice` (not one file per subject). Example doc id: `Conner_Kent__Nov_Dec`.

- Export stamp: `corpus_manifest.yaml` with `corpus_version: 2` (official runs always re-export).
- Train+stable index: ~**1004** unique `doc_id`s (~1550 raw rows; duplicates dropped with logged stats).
- **Not comparable** to P0-TW-01’s ~329-subject deduped index without re-baselining Wave A parity.

## Metrics (eval rows)

| Metric | Meaning |
|--------|---------|
| `retrieval_recall@k` | Gold **subject** appears in any top-k hit |
| `temporal_recall@k` | Top-k hit with matching subject and `last_edited ==` question as-of |
| `answer_em` | Exact match to gold `object` |
| `answer_cosine` | Mean cosine similarity (MiniLM) vs gold `object` |
| `answer_cosine_hit_rate` | Fraction of rows with cosine ≥ 0.85 |

## Parity targets (Wave A, memory BM25)

| Cell | Expect ~ |
|------|----------|
| `train` + `test` | P0-TW-01 → **0.93** recall@5 |
| `train_stable` + `stable` | P0-TW-01b → **0.76** recall@5 |

## Artifacts

- `corpus_train/`, `corpus_train_stable/` — exported stores
- `matrix_results.tsv` — one row per cell
- `agent_traces/<cell_id>/` — per-row traces (Wave B)
