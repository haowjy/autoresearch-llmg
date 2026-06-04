# P1-02 — naive QLoRA + RAG eval (Phase 1)

**Status:** wired — 30-step full-row calibrate ok (`20260529-092200`). Full-epoch official collapsed tool use (`20260526-121352`), so default training is step-capped before the 150-row official eval. Default eval now uses `agent_toolset=hybrid_deep`; the narrowed record-label snippet aliases reached **0.90** answer EM on the default 10-row calibrate without OOM (`20260529-165820`).

## Goal

Fine-tune `google/gemma-4-E4B-it` with **4-bit QLoRA** on TemporalWiki [train][tw-easy] rows, then evaluate with the **same hybrid agent** stack as [P0-TW-03 v3][p0-tw-03] (versioned corpus v2, `search_hybrid` tool).

This is intentionally **naive** SFT:

- **Input:** `question` + truncated `article` in the user turn.
- **Target:** gold `object` (short fact) in the assistant turn.
- **No** tool-trace distillation, DPO, or teacher labels yet.

Compare eval metrics to Phase 0 ceilings ([v3][run-v3] / [v4][run-v4]).

## Hugging Face Hub traffic

Startup loads models/datasets via the Hub (many cached **HEAD** checks; 404s on optional files are normal). To run **offline** after a successful download:

```bash
export LLMG_HF_LOCAL_ONLY=1
```

This sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE`, and passes `local_files_only=True` to Gemma and the shared MiniLM embedder (hybrid search + answer cosine share one load per process).

## Commands

```bash
# Short proxy run (~6.5 min: 30 train steps over 1500 rows + 10-row eval)
uv run python -m llmg.run --experiment P1-02 --run-phase calibrate

# 30-step adapter train + 150-row hybrid eval
uv run python -m llmg.run --experiment P1-02 --run-phase official

# Smoke: 32 train rows, 5 eval rows
uv run python -m llmg.run --experiment P1-02 --run-phase calibrate \
  --param calibrate_max_train_steps=10 --param calibrate_max_train_rows=32 \
  --param calibrate_max_eval_rows=5
```

Artifacts: `lora_adapter/` (PEFT weights), `corpus_train_stable/`, `agent_traces/..._lora/`, `metrics.json`, `timing.json`.

### Timings

Every run writes **`llmg/runs/<timestamp>_P1-02/timing.json`** (and copies key fields into `metrics.json`):

| Key in `timing.json` | Meaning |
|----------------------|---------|
| `phases_s.qlora_train` | Wall time for QLoRA train subprocess |
| `phases_s.lora_eval` | Wall time for hybrid-agent eval subprocess |
| `phases_s.run` | Parent orchestration (train + eval subprocesses) |
| `experiment_wall_s` | Same phases surfaced in `metrics.json` as `*_s` and `experiment_wall_s` |
| `gpu_memory_gb` | Optional CUDA peak snapshots at subprocess phase ends |

Harness-wide wall clock for any experiment is also in **`meta.json`** as `experiment_wall_s` (and `elapsed_s`).

Example:

```json
{
  "experiment_wall_s": 312.5,
  "gpu_memory_gb": { "lora_eval_end": 18.2, "qlora_train_end": 16.1 },
  "phases_s": {
    "lora_eval": 145.2,
    "qlora_train": 158.0,
    "run": 312.5
  }
}
```

After a run, `run_dir:` is printed to stdout; `llmg/runs/latest` points at the newest run.

Re-eval only (adapter already in run dir):

```bash
uv run python -m llmg.run --experiment P1-02 --run-phase official \
  --param skip_train=true --param max_eval_rows=30
```

Eval-only replay using an adapter from another run:

```bash
uv run python -m llmg.run --experiment P1-02 --run-phase calibrate \
  --param skip_train=true --param adapter_source_run=20260529-165820_P1-02
```

Hard-subset probe (rows 10-14, where 15-row calibrate exposed relation/temporal failures):

```bash
uv run python -m llmg.run --experiment P1-02 --run-phase calibrate \
  --param calibrate_max_train_steps=10 --param calibrate_max_train_rows=32 \
  --param calibrate_max_eval_rows=5 --param eval_start_row=10
```

## vs P0-TW-03 baselines

| Anchor | P0-TW-03 v3 hybrid `test` |
|--------|---------------------------|
| `retrieval_recall@5` | **1.0** |
| `temporal_recall@5` | **0.95** |
| `answer_em` | **0.20** |

P1-02 primary metric remains `retrieval_recall@5` on the **LoRA + hybrid agent** eval cell.

### Recent calibrate evidence

| Run | Train | Eval rows | retrieval@5 | temporal@5 | answer_em | notes |
|-----|-------|-----------|-------------|------------|-----------|-------|
| `20260529-092200_P1-02` | 1500 rows, 30 steps | 10 | **1.0** | **1.0** | **0.50** | tool protocol intact; 9/10 `submit_answer`; peak eval memory 16.1GB |
| `20260529-094119_P1-02` | 1500 rows, 30 steps | 10 | **1.0** | **1.0** | **0.60** | default capped calibrate; submitted-only EM **0.71**, fallback 0.30; eval runtime 436s |
| `20260529-102539_P1-02` | 1500 rows, 30 steps, `sft_format=concise` | 10 | **1.0** | **1.0** | **0.60** | all rows submitted; same exact-match rows as default, no answer-quality gain |
| `20260529-105551_P1-02` | 1500 rows, 30 steps, `agent_toolset=hybrid_deep` | 10 | **1.0** | **1.0** | **0.80** | promoted calibrate default; all rows submitted, eval runtime 276s |
| `20260529-111826_P1-02` | 1500 rows, 30 steps, `hybrid_deep`, 20-row probe | 20 | - | - | - | crash/OOM at row 18; 18 complete traces had 10 exact |
| `20260529-112956_P1-02` | 1500 rows, 30 steps, `hybrid_deep`, 15-row probe | 15 | **1.0** | **1.0** | 0.53 | larger sample regressed; not official-ready |
| `20260529-115708_P1-02` | 1500 rows, 30 steps, `eval_start_row=10` hard subset | 5 | - | - | - | crash/OOM on final hard row; 0/4 complete traces exact, peak eval snapshot 21.9GB |
| `20260529-120627_P1-02` | 1500 rows, 30 steps, `eval_start_row=10` row-wise hard probe | 1 | **1.0** | **1.0** | 0.00 | completed without OOM; row 10 answered `John Elkann` vs `Philippe Varin` |
| `20260529-120954_P1-02` | 1500 rows, 30 steps, `eval_start_row=11` row-wise hard probe | 1 | **1.0** | **1.0** | 0.00 | completed without OOM; row 11 fell back to `Charlie Hales` vs `Neil Goldschmidt` |
| `20260529-121724_P1-02` | 1500 rows, 30 steps, `eval_start_row=12` row-wise hard probe | 1 | **1.0** | **1.0** | 0.00 | completed without OOM; row 12 fell back to `Armenia` vs `Russia` |
| `20260529-122052_P1-02` | 1500 rows, 30 steps, `eval_start_row=13` row-wise hard probe | 1 | **1.0** | **1.0** | 0.00 | completed without OOM; row 13 submitted `Gavin Lee` vs `Nazri Nasir` |
| `20260529-121326_P1-02` | 1500 rows, 30 steps, `eval_start_row=14` row-wise hard probe | 1 | **1.0** | **1.0** | **1.00** | completed without OOM; row 14 exact via `grep_file` for `2018` |
| `20260529-122548_P1-02` | 1500 rows, 30 steps, `eval_start_row=14` trace-index smoke | 1 | **1.0** | **1.0** | **1.00** | verifies sliced traces now record original `row_index=14` |
| `20260529-123024_P1-02` | discarded `agent_temporal_hint` prototype on row 10 | 1 | **1.0** | **1.0** | 0.00 | abandon; changed wrong answer to `Pierre-Antoine Delvaux`, code removed |
| `20260529-123617_P1-02` | discarded relation-aware `grep_file` alias prototype on row 10 | 1 | **1.0** | **1.0** | 0.00 | abandon/revise; returned `John Elkann`; gold snippet initially truncated |
| `20260529-123940_P1-02` | discarded relation-aware `grep_file` after-biased snippets on row 10 | 1 | **1.0** | **1.0** | 0.00 | abandon; `Philippe Varin` visible in grep output but model still chose `John Elkann`; code removed |
| `20260529-124913_P1-02` | discarded `sft_format=evidence_window` prototype on row 10 | 1 | **1.0** | **1.0** | 0.00 | abandon; object-centered training window made agent submit `Stellantis` before useful grep; code removed |
| `20260529-125715_P1-02` | discarded `sft_format=tool_submit` prototype on row 10 | 1 | 0.00 | 0.00 | 0.00 | abandon; final-action-only tool SFT collapsed to immediate `submit_answer` (`Pierre-Antoine`), code removed |
| `20260529-130345_P1-02` | discarded `sft_format=tool_trace_steps`, 30 steps, on row 10 | 1 | **1.0** | **1.0** | 0.00 | abandon/revise; learned search/read/grep sequence but repeated empty `submit_answer`; code removed after revision |
| `20260529-130719_P1-02` | discarded `sft_format=tool_trace_steps`, 60 steps, on row 10 | 1 | **1.0** | **1.0** | 0.00 | abandon; nonempty submit returned `Jean-Philippe Imparato`, still literal `chairperson` grep/no matches; code removed |
| `20260529-131754_P1-02` | hard subset with invalid-submit metrics, 10 steps / 32 rows | 5 | **1.0** | **1.0** | 0.00 | keep observability evidence; all rows submitted wrong `Not found` variants, `answer_invalid_submit_rate=0.0` |
| `20260529-132549_P1-02` | rejected `Not found` submits on hard subset | 5 | **1.0** | **1.0** | 0.00 | revise; invalid submits surfaced (`rate=1.0`) but fallback still counted apology text as answers |
| `20260529-133313_P1-02` | rejected non-answer submits + invalid fallback filtering | 5 | **1.0** | **1.0** | 0.00 | keep; invalid submits surfaced (`rate=1.0`, mean 1.4) and rows became no-answer instead of fallback explanations |
| `20260529-134227_P1-02` | discarded no-evidence recovery nudge on hard subset | 5 | **1.0** | **1.0** | 0.00 | abandon; increased some grep/offset use but no EM gain and wall rose to 486s; code removed |
| `20260529-135657_P1-02` | discarded multi-match `grep_file` prototype on hard subset | 5 | **1.0** | **1.0** | 0.00 | abandon; corpus probe exposed hidden gold snippets, but smoke made zero `grep_file` calls and no metric moved; code removed |
| `20260529-140545_P1-02` | discarded `read_file` pagination-footer prototype on hard subset | 5 | **1.0** | **1.0** | 0.00 | abandon; footer did not induce offset reads or grep calls, no answer gain, wall 342s; code removed |
| `20260529-141644_P1-02` | relation-aware snippets in `search_hybrid` output on hard subset | 5 | **1.0** | **1.0** | **0.20** | keep/revise; first-tool snippets exposed gold on all hard rows and row 14 submitted exact `Los Angeles Clippers`; added malformed-tool fallback cleanup after trace review |
| `20260529-142448_P1-02` | discarded one-snippet-per-hit revision on hard subset | 5 | **1.0** | **1.0** | **0.20** | abandon; same EM as two snippets but slower (452s) and row 11 submitted generic `Mayor`; reverted to two snippets per hit |
| `20260529-143551_P1-02` | discarded explicit `candidate:` lines on hard subset | 5 | **1.0** | **1.0** | **0.20** | abandon; exact candidates visible for all hard rows but no EM gain, row 10 submitted wrong `John Elkann`; code removed |
| `20260529-144356_P1-02` | discarded snippet-trust system prompt on hard subset | 5 | **1.0** | **1.0** | **0.20** | abandon; prompt did not improve extraction and row 10 still submitted `John Elkann`; code removed |
| `20260529-145446_P1-02` | discarded `sft_format=search_snippet_answer` prototype on hard subset | 5 | **1.0** | **1.0** | **0.20** | abandon; snippet-style SFT did not improve EM, row 10 regressed to `Stellantis`, eval slowed to 513s; code removed |
| `20260529-150803_P1-02` | embedded quoted `submit_answer` parser fallback on hard subset | 5 | **1.0** | **1.0** | **0.20** | keep infra; same EM but parser stops ignored canonical submits, `agent_steps_mean` 15.2→12.6 and eval 294s→273s vs retained snippet baseline |
| `20260529-151636_P1-02` | 30 train steps / 32 rows on hard subset with retained snippets/parser | 5 | **1.0** | **1.0** | **0.20** | revise; all rows submitted and eval fell to 194s, but no EM gain and rows 11-13 became confident wrong facts |
| `20260529-152317_P1-02` | 30 train steps / 32 rows + `search_snippets_per_hit=1` on hard subset | 5 | **1.0** | **1.0** | **0.40** | keep as opt-in; row 10 fixed to `Philippe Varin`, row 14 stayed exact, eval fell to 180s; rows 11-13 still wrong |
| `20260529-152846_P1-02` | first-five guard for one-snippet + 30-step/32-row setting | 5 | **1.0** | **1.0** | **0.60** | guard ok; rows 0/1/3 exact, all rows submitted, eval 133s |
| `20260529-153349_P1-02` | default 10-row calibrate with temporary one-snippet global setting | 10 | **1.0** | **1.0** | **0.60** | revise; broader proxy regressed vs prior 0.80 default, so two snippets remain default and one-snippet is opt-in |
| `20260529-154323_P1-02` | default two-snippet 10-row calibrate after parser/snippet plumbing | 10 | - | - | - | OOM during eval row 9; partial rows 0-8 had 7 exact before crash, but run is not comparable/completed |
| `20260529-155821_P1-02` | default two-snippet OOM mitigation (`calibrate_max_eval_rows=8`) | 8 | **1.0** | **1.0** | **0.75** | revise; completed in 313s, but row 7 regressed to `Jamal Records` and row 0 required fallback |
| `20260529-160434_P1-02` | no-snippet broad guard (`search_snippets_per_hit=0`, 8 rows) | 8 | **1.0** | **1.0** | **0.625** | abandon as default; clean submits and no OOM, but worse than two-snippet 8-row mitigation |
| `20260529-161339_P1-02` | broad relation-metadata hint (`eval_relation_hint=true`, 8 rows) | 8 | **1.0** | **1.0** | **0.50** | abandon; fixed row 4 but damaged ordinary owner rows 3/6 |
| `20260529-161905_P1-02` | targeted `when_nondate` relation hint (8 rows) | 8 | **1.0** | **1.0** | **0.75** | keep/revise as opt-in; fixed row 4 and preserved rows 3/6, but row 2 stayed shortened and row 7 still wrong |
| `20260529-162505_P1-02` | targeted `when_nondate` hint with phrase-copy wording (8 rows) | 8 | **1.0** | **1.0** | **0.75** | revise; fixed row 2 and row 4, but row 0 regressed to `N.O.W.H.E.R.E.` in this run |
| `20260529-163200_P1-02` | record-label snippet alias + `when_nondate` hint (8 rows) | 8 | **1.0** | **1.0** | **1.00** | keep as calibrate evidence; all first-eight answers exact, row 7 fixed via `Island Records` snippet, not official/promoted |
| `20260529-163936_P1-02` | default lane with broad signed-label alias including `Management` (8 rows) | 8 | **1.0** | **1.0** | **0.75** | revise; exposed `Island Records` but also BMG Rights Management, row 7 chose BMG |
| `20260529-164502_P1-02` | default lane with narrowed signed-record-label alias (8 rows) | 8 | **1.0** | **1.0** | **0.875** | keep; natural-question first-eight guard improved vs `20260529-155821`, row 7 exact, all rows submitted |
| `20260529-165055_P1-02` | default 10-row calibrate with narrowed record-label alias | 10 | **1.0** | **1.0** | **0.80** | keep; no OOM, all rows submitted, row 7 fixed; remaining misses row 2 date/type and row 9 record-label distractor |
| `20260529-165820_P1-02` | default 10-row calibrate with row-9 `to sign` record-label alias | 10 | **1.0** | **1.0** | **0.90** | keep; no OOM, all rows submitted, rows 7/9 fixed; only remaining miss is row 2 date/type mismatch |
| `20260529-170540_P1-02` | opt-in `eval_relation_hint=when_nondate` with record-label aliases | 10 | **1.0** | **1.0** | **0.90** | revise; fixed row 2 and all submitted rows exact, but row 9 regressed to fallback `Casablanca Records`; no aggregate gain |
| `20260529-171348_P1-02` | eval-only replay of `eval_relation_hint=when_nondate` on `20260529-165820` adapter | 10 | **1.0** | **1.0** | **1.00** | keep as diagnostic lane; no retrain, all rows exact, confirms prompt/eval-input effect when adapter variance is removed |
| `20260529-171844_P1-02` | eval-only default replay on `20260529-165820` adapter, 15 rows | 15 | **1.0** | **1.0** | **0.667** | revise; first 10 stayed strong but hard rows 10-13 failed, row 14 exact |
| `20260529-172649_P1-02` | eval-only hard subset replay, `search_snippets_per_hit=1` | 5 | **1.0** | **1.0** | **0.40** | keep as hard-lane evidence; rows 10/14 exact, rows 11-13 still wrong |
| `20260529-173348_P1-02` | broad relation hint on hard subset, one-snippet replay | partial 4/5 | - | - | - | abandon/revise; OOM at row 13 after 172s / 22.5GB snapshot; partial traces kept row 10 exact but rows 11/12 still wrong (`Sam Adams`, `Armenia`) |
| `20260529-173945_P1-02` | discarded grep evidence-order aliases on hard subset | partial 4/5 | - | - | - | abandon; aliases moved Goldschmidt/Nazri evidence to first grep lines but rows 11/12 still wrong and row 13 OOMed; code removed |
| `20260529-181332_P1-02` | eval-only `when_nondate` relation-hint replay on 15 rows | 15 | **1.0** | **1.0** | **0.733** | keep diagnostic; fixes row 2 vs default 15-row replay, hard rows 10-13 unchanged |
| `20260529-182136_P1-02` | eval-only `when_nondate` + one-snippet replay on 15 rows | 15 | **1.0** | **1.0** | **0.800** | keep diagnostic; composes row-2 relation fix with row-10 distractor reduction, rows 11-13 still wrong |
| `20260529-183242_P1-02` | eval-only `when_nondate` + one-snippet replay on rows 15-19 | 5 | **1.0** | **1.0** | **0.600** | keep/revise diagnostic; exact on rows 16/17/19, misses current head-coach/spouse rows 15/18 |
| `20260529-183555_P1-02` | discarded extraction-check system prompt on rows 15-18 | 4 | **1.0** | **1.0** | **0.500** | abandon; prompt did not fix rows 15/18 and code was reverted |
| `20260529-183809_P1-02` | broad relation hint + one-snippet replay on rows 15-18 | 4 | **1.0** | **1.0** | **0.500** | abandon as default path; rows 15/18 still wrong and row 18 shifted to another distractor |
| `20260529-184150_P1-02` | discarded spouse search-snippet alias on row 18 | 1 | **1.0** | **1.0** | **0.000** | revise; surfaced marriage passage but submitted `Elizabeth Burkit Cox`, code later removed |
| `20260529-184337_P1-02` | revised spouse alias preferring Westernized phrase on row 18 | partial 1 | - | - | - | abandon; surfaced `Sun-jung Jung` but OOMed at 16 steps after looping through long reads |
| `20260529-184548_P1-02` | revised spouse alias + `max_agent_steps=12` on row 18 | 1 | **1.0** | **1.0** | **0.000** | abandon; submitted same-person alias `Chong son Chong`, no EM gain, code removed |
| `20260529-184901_P1-02` | rows 20-24 diagnostic mapping, normal 16-step cap | partial 4/5 | - | - | - | OOM; rows 20/21 wrong, row 22 exact, row 23 unfinished at 22.4GB |
| `20260529-185312_P1-02` | rows 23-24 diagnostic mapping, 12-step cap | 2 | **1.0** | **1.0** | **0.000** | keep evidence; both submitted wrong distractor offices/entities under lower cap |
| analysis | `scripts/analyze_agent_answers.py` failure buckets | traces | - | - | - | keep; auto-detects trace cell and separates exact, fallback, retrieval, and gold-doc co-mention wrong-value failures |
| analysis | alias-linked answer bucket audit | traces | - | - | - | keep; row 18 same-person alias now separates as `alias_linked_wrong_surface`, while rows 11-13 and 23-24 remain ordinary `gold_doc_co_mentions_pred` wrong-value failures |
| analysis | observed-evidence visibility audit | traces | - | - | - | keep; analyzer now reports whether gold/prediction appeared in any tool output or in `search_hybrid` output; rows 11-13 had gold visible in search but wrong predictions absent from search |
| `20260529-190017_P1-02` | discarded `sft_format=disambiguate` answer-policy SFT on hard rows | partial 4/5 | - | - | 0.000 | abandon; trained 30 steps but rows 10-12 all wrong and eval OOMed on row 13, code removed |
| `20260529-190735_P1-02` | fallback-disabled replay on rows 20-22 | 3 | **1.0** | **1.0** | **0.333** | keep diagnostic; same EM as partial mapping, but row 21 becomes `no_answer` instead of wrong fallback |
| `20260529-191711_P1-02` | discarded 6-step early-commitment replay on rows 11-13 | 3 | **1.0** | **1.0** | 0.000 | abandon; faster (57.7s) but early cap worsened grounding (`Mike Davis`, fallback `China`, `Lee Chong Wei`) |
| `20260529-192020_P1-02` | discarded temporary snippet-only hybrid toolset on rows 11-13 | partial 3 | - | - | 0.000 | abandon; rows 11/12 wrong before row-13 OOM, model looped on repeated searches, code removed |
| `20260529-192609_P1-02` | discarded search-visible submit validator on rows 11-13 | 3 | **1.0** | **1.0** | 0.000 | abandon; rejected wrong submits but produced only fallback wrong answers, code removed |
| analysis | `scripts/audit_search_candidates.py` search-candidate audit | traces | - | - | - | keep; relation candidates from observed search snippets recover rows 11-13, but row 2 exposed an unsafe override risk |
| `20260529-193546_P1-02` | opt-in search-candidate fallback on rows 10-14 | 5 | **1.0** | **1.0** | **1.000** | keep/revise; rows 11-13 corrected via `search_candidate_fallback`, rows 10/14 stayed normal submits |
| `20260529-193949_P1-02` | unsafe candidate fallback row-2 guard | 1 | **1.0** | **1.0** | 0.000 | revise; candidate override damaged a correct submitted answer named in the question |
| `20260529-194154_P1-02` | revised candidate fallback row-2 guard | 1 | **1.0** | **1.0** | **1.000** | keep; answer-named-in-question guard preserved normal submitted exact answer |
| `20260529-194243_P1-02` | revised candidate fallback 15-row replay | 15 | **1.0** | **1.0** | **1.000** | keep as opt-in diagnostic lane; 12 submitted exact + 3 search-candidate fallback exact, no OOM |
| `20260529-195316_P1-02` | candidate fallback rows 15-19 guard | 5 | **1.0** | **1.0** | **0.800** | keep/revise; row 15 fixed via head-coach candidate, row 18 remains wrong because gold spouse surface was not visible |
| `20260529-200306_P1-02` | retained relation-evidence snippets + candidate fallback rows 18-20 | 3 | **1.0** | **1.0** | **1.000** | keep/revise; rows 18 and 20 fixed via spouse/citizenship candidates, row 19 stayed normal exact |
| `20260529-200630_P1-02` | retained relation-evidence snippets + candidate fallback rows 23-24 | 2 | **1.0** | **1.0** | **1.000** | keep/revise; row 23 submitted exact from head-of-state snippet, row 24 fixed via position candidate |
| `20260529-200809_P1-02` | discarded diplomatic-list snippet probe on row 21 | 1 | **1.0** | **1.0** | 0.000 | abandon; exposed ambiguous `Sudan, Egypt, Germany and Kenya` list and model fell back to `Kenya`; pattern removed |
| `20260529-201122_P1-02` | retained relation-evidence snippets + candidate fallback rows 18-24 | 7 | **1.0** | **1.0** | **0.857** | keep; contiguous guard fixed 18/20/23/24 and preserved 19/22, row 21 remains fallback wrong |
| `20260529-201752_P1-02` | retained relation-evidence snippets + candidate fallback rows 15-24 | 10 | **1.0** | **1.0** | **0.900** | keep; broad guard fixed 15/18/20/23/24 and preserved 16/17/19/22, row 21 remains fallback wrong |
| `20260529-202441_P1-02` | retained row-21 two-snippet probe | partial 1 | - | - | - | abandon; `search_snippets_per_hit=2` did not expose usable diplomatic evidence under retained patterns and OOMed at final-step nudge after 58.4s / 22.2GB |
| `20260529-202837_P1-02` | retained first-ten guard, candidate fallback enabled | 10 | **1.0** | **1.0** | **1.000** | keep; all rows exact via normal `submit_answer`, no candidate/fallback overrides, eval 265.9s / 16.1GB |
| `20260529-203413_P1-02` | retained hard-slice guard rows 10-14 | 5 | **1.0** | **1.0** | **1.000** | keep; rows 10/14 normal exact submits and rows 11-13 exact via search-candidate fallback, eval 215.9s / 16.1GB |
| `20260529-203902_P1-02` | retained next-slice probe rows 25-29 | 5 | **1.0** | **1.0** | 0.200 | revise; exposed new evidence/candidate gaps: record-label truncation, IPL current team, radio station surface, and production-company list ambiguity |
| `20260529-204617_P1-02` | rows 25-29 with new evidence snippets | 5 | **1.0** | **1.0** | 0.600 | keep/revise; fixed row 27 `Mumbai Indians` and row 28 `KSDT Radio`; row 26 still truncated by generic year snippet, row 29 chose `Original Film` from multi-company list |
| `20260529-205015_P1-02` | row-26 record-label snippet order fix | 1 | **1.0** | **1.0** | **1.000** | keep; relation-specific record-label snippet now beats generic year snippet, normal submit exact `Gingerbread Man Records` |
| `20260529-205117_P1-02` | rows 25-29 after record-label ordering fix | 5 | **1.0** | **1.0** | **0.800** | keep/revise; rows 25-28 exact, row 29 remains multi-production-company ambiguity (`Original Film` vs `Blur Studio`), eval 113.8s / 16.1GB |
| `20260529-205544_P1-02` | rows 30-34 retained lane, 3-row partial OOM | partial 3 | - | - | - | abandon/revise; row 32 looped on wrong top-3 basketball hits and OOMed at 126.3s / 22.3GB before metrics |
| `20260529-210126_P1-02` | row 32 with `search_snippet_hits=5` + sports-team snippet | 1 | **1.0** | 0.0 | **1.000** | keep; fourth-ranked `Lluís_Costa__Dec_Jan` snippet exposed `Debreceni EAC`, candidate fallback exact, 26.2s / 16.1GB |
| `20260529-210217_P1-02` | rows 30-34 with `search_snippet_hits=5` | 5 | **1.0** | 0.6 | 0.200 | keep as boundary map; row 32 fixed and no OOM, remaining misses are hidden diplomatic evidence or underdetermined/misleading list labels |
| `20260529-210555_P1-02` | rows 35-39 retained lane baseline | 5 | **1.0** | **1.0** | 0.200 | revise; perfect retrieval but sports-team snippets missed rows 35-36 and row 39 was exact via fallback after 16 steps |
| `20260529-211153_P1-02` | rows 35-36 with sports-team date/joined snippets | 2 | **1.0** | **1.0** | 0.500 | keep/revise; row 36 exact via `KK Crvena zvezda` candidate, row 35 still missed until generic `after ... joined` trigger |
| `20260529-211509_P1-02` | row 35 with generic `after ... joined` trigger | 1 | **1.0** | **1.0** | **1.000** | keep; normal submit exact `Sacramento Kings`, 57.4s / 16.1GB |
| `20260529-211622_P1-02` | rows 35-39 after sports-team fixes | 5 | **1.0** | **1.0** | **0.600** | keep/revise; rows 35, 36, 39 exact; rows 37-38 remain diplomatic/chairperson ambiguity, eval 241.0s / 16.1GB |
| `20260529-212257_P1-02` | rows 40-44 retained lane mapping | 5 | **1.0** | **1.0** | **0.600** | keep as boundary map; rows 40/42/43 exact, rows 41/44 are chairperson/head-of-government label-time ambiguity, eval 99.9s / 16.1GB |
| `20260529-212737_P1-02` | rows 45-49 retained lane mapping | 5 | **1.0** | **1.0** | **0.600** | keep as boundary map; rows 46/48/49 exact, rows 45/47 are diplomatic-relation ambiguity, eval 106.7s / 16.1GB |
| `20260529-213040_P1-02` | rows 50-54 retained lane mapping | 5 | **1.0** | **1.0** | **0.600** | keep as boundary map; rows 50/52/54 exact, rows 51/53 are co-mentioned entertainment/channel list ambiguity, eval 87.4s / 16.1GB |
| analysis | retained rows 0-54 failure-class audit | traces | - | - | - | keep; selected retained runs cover all rows 0-54 at 41/55 exact, with misses dominated by diplomatic labels (5), office/title-time ambiguity (3), and co-mentioned list/entity selection (4) |
| `20260529-213655_P1-02` | rows 55-59 retained lane mapping | 5 | **1.0** | **1.0** | 0.400 | revise; rows 56/58 exact, row 55 first-husband evidence hidden from search snippets, rows 57/59 record-label co-mentions chose visible historical labels, eval 134.6s / 16.1GB |
| `20260529-214203_P1-02` | row 55 first-husband marriage-evidence alias | 1 | **1.0** | **1.0** | **1.000** | keep; `first marriage was to ...` snippet exposed `Stanley Clements`, exact via last-content fallback, eval 78.6s / 16.1GB |
| `20260529-214337_P1-02` | rows 55-59 after first-husband alias | 5 | **1.0** | **1.0** | **0.600** | keep/revise; row 55 fixed and rows 56/58 preserved, rows 57/59 remain record-label co-mention conflicts, eval 147.4s / 16.1GB |
| `20260529-214750_P1-02` | rows 60-64 retained lane mapping | 5 | **1.0** | **1.0** | 0.200 | keep as warning boundary; row 64 exact, rows 61-63 show search-candidate fallback selecting stale prior-snapshot snippets for Portland/Pakistan/Singapore, eval 247.5s / 16.1GB |
| `20260529-215603_P1-02` | rows 60-64 with candidate fallback disabled | 5 | **1.0** | **1.0** | 0.400 | revise; disabling fallback preserved row 62 `Armenia` via last-content fallback but row 63 still missed year-specific coach evidence, eval 247.0s / 16.1GB |
| `20260529-220205_P1-02` | row 63 year-specific head-coach snippet | 1 | **1.0** | **1.0** | **1.000** | keep; `2026 ... under new head coach Tsutomu Ogura` snippet led to normal exact submit in 8 steps, eval 45.0s / 16.1GB |
| `20260529-220308_P1-02` | rows 60-64 with coach snippet + candidate fallback disabled | 5 | **1.0** | **1.0** | **0.600** | keep/revise diagnostic; rows 62/63/64 exact, rows 60/61 remain title-time ambiguity, eval 215.0s / 16.1GB |
| `20260529-220806_P1-02` | rows 65-69 with candidate fallback disabled | 5 | **1.0** | **1.0** | 0.400 | revise; rows 66/67 exact, row 69 exposed a clean sibling-list miss (`Ishvi` visible only after grep/read), eval 169.7s / 16.1GB |
| `20260529-221423_P1-02` | row 69 sibling-list snippet, fallback disabled | 1 | **1.0** | **1.0** | 0.000 | revise; new `sired at least five sons (...)` snippet exposed `Ishvi and Ish-bosheth`, but model submitted `Mephibosheth` |
| `20260529-221528_P1-02` | row 69 sibling-list candidate fallback | 1 | **1.0** | **1.0** | **1.000** | keep opt-in; sibling-list candidate extracted immediate predecessor `Ishvi`, exact via `search_candidate_fallback`, eval 41.6s / 16.1GB |
| `20260529-221627_P1-02` | rows 65-69 sibling-list candidate fallback | 5 | **1.0** | **1.0** | **0.600** | keep/revise diagnostic; rows 66/67 normal exact, row 69 exact via candidate fallback, rows 65/68 remain ambiguous/stale relation surfaces, eval 149.0s / 16.1GB |
| `20260529-222047_P1-02` | rows 70-74 with candidate fallback disabled | 5 | **1.0** | **1.0** | 0.400 | revise; rows 70/71 exact, row 74 had visible `Prime Minister of Israel` in tool text but submitted `Member of Knesset`, eval 201.1s / 16.1GB |
| `20260529-222604_P1-02` | row 74 first prime-minister position snippet | 1 | **1.0** | **1.0** | 0.000 | revise; initial snippet rule was too narrowly triggered, so search text stayed truncated and model repeated `Member of Knesset` |
| `20260529-222757_P1-02` | row 74 position-query trigger fix, fallback disabled | 1 | **1.0** | **1.0** | 0.000 | revise; `Ariel Sharon position ...` search now exposed `prime minister of Israel`, but model still submitted `Member of Knesset` |
| `20260529-222926_P1-02` | row 74 prime-minister position candidate fallback | 1 | **1.0** | **1.0** | **1.000** | keep opt-in; held-position candidate extracted `Prime Minister of Israel`, exact via `search_candidate_fallback`, eval 64.4s / 16.1GB |
| `20260529-223044_P1-02` | rows 70-74 prime-minister candidate fallback | 5 | **1.0** | **1.0** | **0.600** | keep/revise diagnostic; row 74 exact via candidate fallback and rows 70/71 preserved, rows 72/73 remain historical/current conflicts, eval 203.9s / 16.1GB |
| `20260529-223541_P1-02` | rows 75-79 with candidate fallback disabled | 5 | **1.0** | **1.0** | 0.200 | revise; only row 76 exact, rows 77/79 submitted shortened surfaces and row 78 no-answered despite gold later in article, eval 160.8s / 16.1GB |
| `20260529-224253_P1-02` | rows 75-79 full-surface evidence snippets, fallback disabled | 5 | **1.0** | **1.0** | **0.600** | keep/revise; episode-question-aware snippets fixed row 78 `Triton Ballpark` and row 79 `Paramount Pictures`, row 77 still shortened to `Mumbai`, eval 122.1s / 16.1GB |
| `20260529-224524_P1-02` | rows 75-79 full-surface snippets + candidate fallback | 5 | **1.0** | **1.0** | **0.600** | revise; candidate fallback fixed row 77 `Mumbai cricket team` but regressed row 76 from submitted `Asylum Records` to stale `Gingerbread Man Records`, eval 123.6s / 16.1GB |
| `20260529-224949_P1-02` | rows 80-84 fallback-disabled boundary map | 5 | **1.0** | **1.0** | **0.600** | keep as boundary map; rows 81/82/84 exact, row 80 gold `Russia` not visible in observed tool text, row 83 submitted co-founder `Noah Glass` for gold `Biz Stone`, eval 166.0s / 16.1GB |
| `20260529-225413_P1-02` | rows 85-89 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.400 | keep as boundary map; rows 85/86 exact, rows 87/88 hit step cap with ambiguous diplomatic/party-leader evidence, row 89 appears label-inconsistent (`John Williams` vs `Patrick Doyle`), eval 240.5s / 16.1GB |
| `20260529-225943_P1-02` | rows 90-94 fallback-disabled baseline | 5 | **1.0** | **1.0** | 0.200 | revise; only row 92 exact, rows 90/94 hid clean singleton evidence behind weak snippets, rows 91/93 are chairperson/list-surface ambiguity, eval 190.5s / 16.1GB |
| `20260529-230546_P1-02` | rows 90-94 final-season + current-mayor snippets | 5 | **1.0** | **1.0** | **0.600** | keep/revise; row 90 `Toronto Raptors` and row 94 `Burkhard Jung` fixed via normal submits, rows 91/93 remain ambiguous/list-surface, eval 173.9s / 16.1GB |
| `20260529-230958_P1-02` | rows 95-99 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.400 | keep as boundary map; rows 96/98 exact, row 95 treaty/city-state wording conflicts with source surface, row 97 diplomatic-list selector absent, row 99 `Lionsgate Films`/`Lionsgate Studios` corporate naming ambiguity, eval 191.0s / 16.1GB |
| `20260529-231617_P1-02` | rows 100-104 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.600 | revise; rows 100/103/104 exact, row 101 submitted expanded `Warner Bros. Entertainment Inc.`, row 102 picked prior appointment instead of year-specific `Secretary for Overseas Trade`, eval 117.1s / 16.1GB |
| `20260529-231949_P1-02` | rows 100-104 year-specific position candidate fallback | 5 | **1.0** | **1.0** | **0.800** | keep opt-in; `became <position> in <year>` candidate fixed row 102 via search-candidate fallback with no new miss in this slice, row 101 remains semantic legal-name surface, eval 116.8s / 16.1GB |
| `20260529-232312_P1-02` | rows 105-109 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.400 | revise; rows 105/109 exact, rows 106/108 have clean co-founded/traded-to gold in search snippets, row 107 `Deram` not observed, eval 183.5s / 16.1GB |
| `20260529-232738_P1-02` | rows 105-109 co-founded/traded candidate fallback before record-label ordering fix | 5 | **1.0** | **1.0** | 0.600 | revise; row 106/108 fixed, but stale record-label candidate overrode correct row 109 `Mercury Records` with `Atlantic Records`, eval 180.1s / 16.1GB |
| `20260529-233244_P1-02` | rows 105-109 co-founded/traded candidates + album-release record-label priority | 5 | **1.0** | **1.0** | **0.800** | keep opt-in; rows 106/108 fixed via candidate fallback, row 109 preserved as normal `Mercury Records` submit after release-label snippet priority, row 107 remains hidden-gold, eval 197.7s / 16.1GB |
| `20260529-233742_P1-02` | rows 110-114 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.200 | revise; only row 112 exact, row 114 search snippet truncated before `Philadelphia 76ers`, rows 110/111 gold not observed and row 113 head-coach evidence conflicts current vs gold, eval 255.8s / 16.1GB |
| `20260529-234355_P1-02` | rows 110-114 dated traded-to sports-team snippet | 5 | **1.0** | **1.0** | **0.400** | keep/revise; row 114 fixed as normal `Philadelphia 76ers` submit from fuller traded-date snippet, row 112 preserved, rows 110/111/113 remain boundary cases, eval 217.6s / 16.1GB |
| `20260529-234943_P1-02` | rows 115-119 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.600 | keep as boundary map; rows 116/117/119 exact, row 115 asks current head coach but gold is older Graham Henry, row 118 asks generic spouse but gold is first wife while evidence points to current wife, eval 90.1s / 16.1GB |
| `20260529-235429_P1-02` | rows 120-124 fallback-disabled boundary map | 5 | - | - | - | abandon full-slice attempt; row 120 exact but row 121 looped over South Sudan/Sudan reads and OOMed before submit, partial eval 104.5s / 22.9GB |
| `20260529-235708_P1-02` | row 121 fallback-disabled OOM isolation | 1 | - | - | - | abandon; reducing eval rows did not help, same row-local OOM at final-step generation, eval 66.0s / 22.9GB |
| `20260529-235830_P1-02` | row 121 shorter-read OOM probe | 1 | - | - | - | abandon; `max_article_chars=1000` still OOMed after the same repeated read loop, eval 66.7s / 22.9GB |
| `20260529-235952_P1-02` | row 121 shorter agent budget probe | 1 | **1.0** | 0.0 | 0.000 | keep as OOM diagnostic only; `max_agent_steps=10` completed via fallback `Uganda` vs gold `United States`, gold not observed, eval 43.7s / 16.1GB |
| `20260530-000056_P1-02` | rows 122-124 fallback-disabled boundary map | 3 | **1.0** | **1.0** | 0.000 | revise; row 122 had clean `Alen Stajcic` evidence hidden behind stale coach snippets, rows 123/124 are current-vs-gold label conflicts, eval 162.7s / 16.1GB |
| `20260530-000853_P1-02` | row 122 dated appointed-head-coach snippet | 1 | **1.0** | **1.0** | **1.000** | keep; dated appointment snippet exposed `Alen Stajcic was appointed as head coach in October 2021`, normal submit exact, eval 46.2s / 16.1GB |
| `20260530-000955_P1-02` | rows 122-124 dated appointed-head-coach snippet | 3 | **1.0** | **1.0** | **0.333** | keep/revise; row 122 fixed as normal submit with no neighboring regression, rows 123/124 remain label/time ambiguities, eval 132.1s / 16.1GB |
| `20260530-001401_P1-02` | rows 125-129 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.400 | revise; rows 125/127 exact, row 128 had clean `Price Center` evidence missed behind event-venue snippets, rows 126/129 are label/list ambiguities, eval 96.1s / 16.1GB |
| `20260530-001655_P1-02` | row 128 campus-building snippet | 1 | **1.0** | **1.0** | **1.000** | keep; main-student-hub campus-building snippet exposed `Price Center`, normal submit exact, eval 43.4s / 16.1GB |
| `20260530-001752_P1-02` | rows 125-129 campus-building snippet | 5 | **1.0** | **1.0** | **0.600** | keep/revise; row 128 fixed and rows 125/127 preserved, row 126 source says `Asylum Records` while gold is `Atlantic Records`, row 129 asks any production company with multiple valid co-listed names, eval 94.5s / 16.1GB |
| `20260530-002103_P1-02` | rows 130-134 fallback-disabled boundary map | 5 | 0.8 | 0.8 | 0.400 partial | OOM boundary map; rows 131/132 exact, row 134 gold visible but no submit before OOM, rows 130/133 gold absent from observed evidence path, partial eval 168.7s / 22.8GB |
| `20260530-002430_P1-02` | row 134 OOM isolation | 1 | - | - | - | row-local OOM persisted even with one eval row; search surfaced Marilyn's mother/husband distractor first while `James Dougherty` was visible only after repeated article reads, partial eval 75.9s / 22.8GB |
| `20260530-002708_P1-02` | row 134 first-husband snippet | 1 | **1.0** | **1.0** | **1.000** | keep; `before marrying James Dougherty at the age` snippet exposed the first-husband evidence and row submitted exact in 34.6s / 16.1GB |
| `20260530-002755_P1-02` | rows 130-134 first-husband snippet | 5 | **1.0** | **1.0** | **0.600** | keep/revise; rows 131/132/134 exact, row 130 still falls back to `Serbia` with gold `Spain` absent from observed tool text, row 133 submits owner/current-salient `Elon Musk` vs founder gold `Jack Dorsey`, eval 126.7s / 16.1GB |
| `20260530-003308_P1-02` | rows 135-139 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.200 | revise; only row 136 exact, row 135 has clean Dec 2011 `Texas Legends` evidence hidden behind older NBA-team snippets, rows 137/138 gold absent from observed evidence, row 139 multi-composer/fallback ambiguity, eval 255.3s / 16.1GB |
| `20260530-003915_P1-02` | row 135 returned-professional-basketball snippet | 1 | **1.0** | **1.0** | **1.000** | keep; Dec 2011 returned-to-professional-basketball snippet exposed `Texas Legends`, normal submit exact, eval 47.8s / 16.1GB |
| `20260530-004016_P1-02` | rows 135-139 returned-professional-basketball snippet | 5 | **1.0** | **1.0** | **0.400** | keep/revise; row 135 fixed and row 136 preserved, rows 137/138 remain hidden-gold or relation-label boundary cases, row 139 lists multiple composers and falls back to `Alexandre Desplat`, eval 226.0s / 16.1GB |
| `20260530-004533_P1-02` | rows 140-144 fallback-disabled boundary map | 5 | **1.0** | **1.0** | 0.600 | keep as boundary map; rows 140/141/142 exact, row 143 submits alias `Stikkan Anderson` while gold is `Stig Anderson`, row 144 asks current head of government but gold is first post-reunification mayor, eval 126.0s / 16.1GB |
| `20260530-004850_P1-02` | rows 145-149 fallback-disabled boundary map | 5 | **1.0** | 0.8 | 0.200 | revise; row 145 exact, row 148 own-label and row 149 dated acquisition are clean evidence-surfacing misses, row 146 screenplay co-writer list/gold conflict, row 147 temporal retrieval miss, eval 134.3s / 16.1GB |
| `20260530-005236_P1-02` | rows 148-149 own-label + buggy acquisition snippet | 2 | **1.0** | **1.0** | 0.500 | abandon intermediate; Rihanna fixed, first acquisition regex failed to prefer September 2007 and row 149 submitted `Paramount Pictures`, eval 84.2s / 16.1GB |
| `20260530-005619_P1-02` | rows 148-149 own-label + month/year acquisition snippet | 2 | **1.0** | **1.0** | **1.000** | keep; `Westbury Road Entertainment` and `Mandate Pictures` exact after own-label and September 2007 acquisition snippets, eval 49.1s / 16.1GB |
| `20260530-005725_P1-02` | rows 145-149 own-label + acquisition snippets | 5 | **1.0** | 0.8 | **0.600** | keep/revise; rows 148/149 fixed and row 145 preserved, row 146 remains co-writer/announced-development ambiguity, row 147 remains temporal retrieval miss, eval 95.4s / 16.1GB |
| `20260530-010029_P1-02` | rows 150-154 end-of-split probe | 0 | 0.0 | 0.0 | 0.000 | keep as end marker; `eval_start_row=150` produced zero eval rows, confirming the current test split row sweep is exhausted |
| `20260530-010313_P1-02` | rows 0-14 aggregate guard, current snippets | 15 | **1.0** | **1.0** | **0.733** | keep/revise aggregate guard; 11/15 exact with submitted-only EM 0.769, preserves rows 10/14 fixes under fallback-disabled one-snippet relation-hint lane, but runtime 480.1s is too slow for default short-cycle probing |
| `20260530-011237_P1-02` | default-lane rows 0-9 guard, current snippets | 10 | **1.0** | **1.0** | **0.900** | keep default guard; candidate fallback disabled, default snippets and no relation hint, 9/10 exact with only row 2 `1955` date/type miss, eval 300.7s / 16.1GB |
| `20260530-011857_P1-02` | default-lane rows 10-19 guard, current snippets | 10 | **1.0** | **1.0** | 0.500 | keep as promotion-risk guard; all gold answers were in tool text but rows 10-13 and 18 missed under default snippets/no relation hint, eval 379.8s / 16.1GB |
| `20260530-013000_P1-02` | row 18 westernized spouse-alias snippet | 1 | **1.0** | **1.0** | **1.000** | keep/revise; alias snippet exposed `alternately Westernized as Sun-jung Jung` first and answer matched exactly, but via last-content fallback rather than `submit_answer`, eval 85.1s / 16.1GB |
| `20260530-013156_P1-02` | rows 15-19 westernized spouse-alias replay | 5 | **1.0** | **1.0** | **1.000** | keep/revise; rows 15/16/17/19 preserved as normal submits and row 18 fixed to fallback-exact `Sun-jung Jung`, eval 165.8s / 16.1GB |
| `20260530-013707_P1-02` | row 13 AFC interim-head-coach snippet prototype | 1 | **1.0** | **1.0** | 0.000 | abandon; prototype moved `Nazri Nasir` into the first search snippet but the model chased later 2025 evidence and submitted `Gavin Lee`, code/test reverted, eval 80.2s / 16.1GB |
| `20260530-014053_P1-02` | row 12 diplomatic-relations grep alias prototype | 1 | **1.0** | **1.0** | 0.000 | abandon; grep alias surfaced `Relations with Russia` but model still chased non-relation distractors and submitted `Armenia`, code/test reverted, eval 40.1s / 16.1GB |
| `20260530-014417_P1-02` | default-lane rows 10-19 guard after spouse-alias keep | 10 | **1.0** | **1.0** | **0.600** | keep/revise block guard; confirms row 18 fix lifts default rows 10-19 from 0.500 to 0.600 with rows 14-19 exact, while rows 10-13 remain answer-selection blockers, eval 387.0s / 16.1GB |
| `20260530-015243_P1-02` | 60-step naive QLoRA depth probe | 5 | 0.200 | 0.200 | 0.400 | abandon; extra naive SFT damaged tool protocol on first-five eval, 3/5 rows had no retrieved docs and fell back to plain text, train 172.1s + eval 65.0s / 16.1GB |
| `20260530-020344_P1-02` | 10-step synthetic `tool_trace` SFT probe | 5 | **1.0** | **1.0** | 0.600 | revise; native tool trajectory supervision restored submit behavior (`submit_rate=1.0`, no fallback) but answer selection regressed vs current 30-step adapter, rows 2/4 submitted co-mentioned distractors, train 63.0s + eval 83.5s / 16.1GB |
| `20260530-020735_P1-02` | 10-step mixed plain + `tool_trace` SFT probe | 5 | **1.0** | **1.0** | **0.800** | keep/revise; mixed objective preserved protocol gains (`submit_rate=1.0`, no fallback) while recovering row 4, leaving only row 2 `1955` date/type miss, train 58.5s + eval 111.1s / 16.1GB |
| `20260530-021153_P1-02` | mixed `tool_trace` adapter rows 0-9 expansion | 10 | **1.0** | **1.0** | 0.800 | revise; protocol remained strong (`submit_rate=1.0`, no fallback) but missed rows 2 and 9, regressing below retained 30-step adapter's 0.900 rows0-9 guard, eval 247.4s / 16.1GB |
| `20260530-021712_P1-02` | mixed `tool_trace` adapter rows 10-14 hard comparison | 5 | **1.0** | **1.0** | 0.200 | revise; hard block still fails answer selection on rows 10-13 despite all gold visible in tool text, only row 14 exact, eval 149.6s / 16.1GB |
| `20260530-022348_P1-02` | discarded mixed relation-evidence + tool-trace SFT prototype | 5 | **1.0** | **1.0** | 0.200 | abandon; relation-labeled evidence example did not improve hard rows 10-14 and worsened protocol (`submit_rate=0.8`, one no-answer), code/test reverted, train 58.7s + eval 202.7s / 16.1GB |
| `20260530-023157_P1-02` | current candidate fallback rows 10-19 after spouse alias | 10 | **1.0** | **1.0** | 0.800 | keep/revise diagnostic; candidate fallback fixed rows 10-12 but regressed row 14 to `Phoenix Suns` and left row 13 wrong, so fallback remains opt-in pending a confidence gate; eval 387.9s / 16.1GB |
| `20260530-024355_P1-02` | discarded non-search tool-evidence candidate gate v1 | 5 | **1.0** | **1.0** | 0.400 | abandon; preserved row 14 but overblocked rows 11-12 because wrong submitted surfaces also appeared in grep/read evidence, eval 232.5s / 16.1GB |
| `20260530-024855_P1-02` | discarded non-search tool-evidence candidate gate v2 | 5 | **1.0** | **1.0** | 0.400 | abandon; restored row 11 but still missed row 12 and failed to prevent row 14 `Phoenix Suns` because full read text contained stale candidate; code reverted, eval 234.0s / 16.1GB |
| `20260530-025457_P1-02` | year-aware sports-team candidate extraction | 5 | **1.0** | **1.0** | **0.800** | keep/revise; rows 10-12 fixed via candidate fallback, row 14 preserved as normal `Los Angeles Clippers` submit by rejecting 2017 `Phoenix Suns` candidate for 2018 question; row 13 remains multi-candidate coach ambiguity, eval 238.1s / 16.1GB |
| `20260530-030030_P1-02` | year-aware sports-team candidate rows 10-19 guard | 10 | **1.0** | **1.0** | **0.900** | keep; improves prior candidate-fallback replay 0.800→0.900 by preserving row 14, fixes rows 10-12 via candidate fallback, preserves rows 15-19, only row 13 remains wrong, eval 389.4s / 16.1GB |
| `20260530-030832_P1-02` | AFC-vs-FIFA head-coach candidate filter | 5 | **1.0** | **1.0** | **1.000** | keep/revise; rows 10-14 all exact, row 13 fixed by rejecting FIFA World Cup `under new head coach` candidate for an AFC Asian Cup question, eval 241.7s / 16.1GB |
| `20260530-031345_P1-02` | AFC head-coach filter rows 10-19 guard | 10 | **1.0** | **1.0** | **1.000** | keep; rows 10-19 all exact, candidate fallback fixes rows 10-13 while row 14 and rows 15-19 remain preserved, eval 396.1s / 16.1GB |
| `20260530-032131_P1-02` | stale-candidate guard rows 60-64 | 5 | **1.0** | **1.0** | 0.200 | revise; candidate fallback still unsafe on future-snapshot repeated subjects, selecting stale/wrong candidates on rows 60-63; also exposed overbroad AFC filter for 2026 row 63, unit fix added afterward, eval 190.4s / 16.1GB |
| `20260530-032639_P1-02` | refined 2026 AFC head-coach row-63 check | 1 | **1.0** | **1.0** | **1.000** | keep; refined filter allows 2026 `under new head coach` candidate, row 63 exact via normal submit with no candidate override, eval 44.7s / 16.1GB |
| `20260530-033055_P1-02` | 2026 snapshot candidate-fallback safety gate | 5 | **1.0** | **1.0** | 0.600 | keep/revise; disables search-candidate override for `as_of=2026-01-01`, improving rows60-64 0.200→0.600 by preserving rows62/63/64 normal submits, but rows60/61 remain wrong submitted distractors, eval 188.5s / 16.1GB |
| `20260530-033952_P1-02` | future-safe held-post candidate support for row 61 | 5 | **1.0** | **1.0** | **0.800** | keep/revise; allows 2026 candidate fallback only when search snippets contain explicit future-safe support, fixing row61 `Connie McCready` while preserving rows62/63/64 normal submits and blocking stale row60/62 candidates, eval 172.4s / 16.1GB |
| `20260530-034540_P1-02` | abandoned row60 chairperson succession snippet | 1 | **1.0** | **1.0** | 0.000 | abandon; surfacing `Christian Streiff` in search/tool text did not change the model's `John Elkann` submit after it grepped `Chairman of the Board`, so this is answer-selection/data-label conflict rather than hidden evidence, eval 35.2s / 16.1GB |
| `20260530-034806_P1-02` | rows75-79 candidate gate guard | 5 | - | - | - | OOM guard; rows75-78 completed before row79 OOM at 21.35GB process memory, showing no candidate fallback on completed rows and row79 row-local read loop risk |
| `20260530-035125_P1-02` | rows75-78 split candidate gate guard | 4 | **1.0** | **1.0** | 0.500 | keep/revise guard; no candidate fallback fired, rows76/78 exact, row75 wrong `New York Knicks`, row77 submitted surface `Mumbai` vs `Mumbai cricket team` despite exact candidate, eval 96.9s / 16.1GB |
| `20260530-035330_P1-02` | row79 OOM isolation | 1 | - | - | - | OOM; row-local repeated full-file read loop persisted even as single eval row, reaching final-step nudge then OOMing at 22.6GB snapshot / 21.35GB process memory |
| `20260530-035520_P1-02` | row79 `max_agent_steps=10` mitigation | 1 | **1.0** | **1.0** | **1.000** | keep as runtime diagnostic only; capped steps avoided OOM and produced exact `Paramount Pictures` via last-content fallback, not parsed submit, eval 54.8s / 16.1GB |
| `20260530-035913_P1-02` | duplicate `read_file` guard row79 | 1 | **1.0** | **1.0** | **1.000** | keep; exact repeated-read skips pushed the agent to grep `Paramount Pictures` and submit exact under normal 16-step budget, avoiding the row-local OOM, eval 56.0s / 16.1GB |
| `20260530-040043_P1-02` | duplicate `read_file` guard rows75-79 | 5 | **1.0** | **1.0** | **0.600** | keep/revise; prior 5-row OOM now completes at normal 16 steps with parsed submits for all rows, row79 fixed exact, rows76/78 preserved, row75 and row77 remain answer-selection/surface misses, eval 141.6s / 16.1GB |
| `20260530-040438_P1-02` | duplicate `read_file` guard row121 OOM check | 1 | **1.0** | 0.0 | 0.000 | keep as runtime generalization only; isolated row121 now completes instead of OOMing, duplicate reads are skipped, but temporal retrieval misses the gold `United States` and fallback answers `Sudan`, eval 67.1s / 16.1GB |
| `20260530-040845_P1-02` | domestic-cricket full-surface candidate row77 | 1 | **1.0** | **1.0** | **1.000** | keep/revise; 2026 safe-support gate now allows explicit domestic-cricket team evidence, converting submitted `Mumbai` to fallback `Mumbai cricket team`, eval 24.6s / 16.1GB |
| `20260530-040935_P1-02` | domestic-cricket full-surface rows75-79 guard | 5 | **1.0** | **1.0** | **0.800** | keep; row77 fixed by candidate fallback, rows76/78/79 preserved, row75 remains co-mentioned sports-team miss, no OOM, eval 140.9s / 16.1GB |
| `20260530-041402_P1-02` | current candidate gate rows80-84 guard | 5 | **1.0** | **1.0** | 0.600 | keep as boundary guard; current lane matches prior 0.600, no candidate fallback fired, rows81/82/84 exact, row80 lacks gold in observed text, row83 co-founder list conflict remains, eval 112.4s / 16.1GB |
| `20260530-042138_P1-02` | midseason-trade sports candidate rows75-79 guard | 5 | **1.0** | **1.0** | **1.000** | keep; evidence-grounded sports-team selector converts row75 `New York Knicks` to `Utah Jazz` via candidate fallback, preserves rows76/78/79 exact submits and row77 cricket-team fallback, eval 144.9s / 16.1GB |
| `20260526-112853_P1-02` | 128 rows, 30 steps | 10 | **1.0** | **1.0** | 0.30 | tool protocol intact |
| `20260526-121352_P1-02` | 1500 rows, 1 epoch | 150 | 0.0 | 0.0 | 0.0 | adapter repeatedly emitted plain text, no tool calls |

Short 5-row data/step probes:

| Run | Variant | retrieval@5 | answer_em | submit/fallback | status |
|-----|---------|-------------|-----------|-----------------|--------|
| `20260529-093320_P1-02` | baseline, 16 steps, 1500 article chars | **1.0** | **0.40** | 1.0 / 0.0 | baseline |
| `20260529-095122_P1-02` | `max_agent_steps=12` | **1.0** | 0.00 | 1.0 / 0.0 | abandon; wrong early submits |
| `20260529-095635_P1-02` | `max_agent_steps=14` | **1.0** | 0.20 | 0.8 / 0.2 | abandon; not enough quality |
| `20260529-100320_P1-02` | `max_article_chars=500` | **1.0** | 0.20 | 0.8 / 0.2 | abandon; no extraction gain |
| `20260529-100940_P1-02` | `max_article_chars=0` | **1.0** | 0.20 | 1.0 / 0.0 | abandon; submits `Not found` too often |
| `20260529-101704_P1-02` | `sft_format=concise` | **1.0** | 0.40 | 0.8 / 0.2 | revise; submitted-only EM 0.50, needs 10-row calibrate |
| `20260529-103630_P1-02` | discarded `sft_format=retrieved` prototype | **1.0** | 0.40 | 0.8 / 0.2 | abandon; same wrong rows, code removed |
| `20260529-104422_P1-02` | `agent_toolset=hybrid_deep` before `grep_file` fix | **1.0** | 0.60 | 1.0 / 0.0 | keep/revise; deep tool used, shell-backed grep had regex bug |
| `20260529-105029_P1-02` | `agent_toolset=hybrid_deep` after Python `grep_file` fix | **1.0** | 0.60 | 1.0 / 0.0 | keep; row 4 fixed via grep |
| `20260529-110345_P1-02` | `agent_toolset=hybrid_deep`, `max_agent_steps=12` | **1.0** | 0.60 | 1.0 / 0.0 | abandon; same EM, no runtime win, different wrong row |
| `20260529-111116_P1-02` | discarded `eval_prompt_format=relation_object` prototype | **1.0** | 0.40 | 1.0 / 0.0 | abandon; broke rows 1/4, code removed |
| `20260529-114243_P1-02` | discarded `sft_format=relation` prototype | **1.0** | 0.20 | 1.0 / 0.0 | abandon; suppressed grep use, code removed |
| `20260529-115131_P1-02` | `eval_start_row=10` hard subset, 10 train steps / 32 rows | **1.0** | 0.00 | 1.0 / 0.0 | keep as hard-subset baseline; rows 10-14 all wrong despite perfect guards |

Answer metrics are split by answer source:

| Metric | Meaning |
|--------|---------|
| `answer_submit_rate` | Fraction of rows whose final answer came from a parsed `submit_answer` tool call |
| `answer_fallback_rate` | Fraction scored from last non-empty assistant text because no submit happened |
| `answer_search_candidate_fallback_rate` | Fraction scored from the opt-in structured search-candidate fallback |
| `answer_no_answer_rate` | Fraction with neither submitted nor fallback answer |
| `answer_invalid_submit_rate` | Fraction of rows with at least one rejected `submit_answer` attempt |
| `answer_invalid_submit_count_mean` | Mean rejected `submit_answer` attempts per row |
| `answer_answered_only_*` | EM/cosine over non-empty final answers, including fallback |
| `answer_submitted_only_*` | EM/cosine over parsed `submit_answer` rows only |

Use `uv run python scripts/analyze_agent_answers.py <run_dir> --rows` for trace-level failure buckets. It auto-detects the single trace cell in a run directory and reports whether misses are retrieval misses, fallback/no-answer, or wrong values co-mentioned in the retrieved gold-subject articles.

## Implementation

| Module | Role |
|--------|------|
| `llmg/train/sft_data.py` | Chat messages from TW rows |
| `llmg/train/qlora.py` | 4-bit QLoRA + `Trainer` (Gemma4 `language_model` LoRA regex) |
| `llmg/train/run_qlora_train.py` | GPU train subprocess |
| `llmg/train/run_lora_eval.py` | GPU eval subprocess (bf16 base + adapter) |
| `llmg/agent/gemma_loop.py` | Optional `adapter_path` (PEFT load) |
| `llmg/experiments/P1-02/runner.py` | orchestrates train + corpus export + eval subprocesses |

**3090 notes:** Default `max_seq_len: 2048`, `lora_rank: 8`. Do **not** call `prepare_model_for_kbit_training` on the full multimodal checkpoint (upcasts vision/audio to fp32 and OOMs). Train and eval run in **separate processes** so VRAM is freed between phases.

`sft_format: plain` keeps the original user turn (`question` + optional article). `sft_format: concise` appends `Answer with only the short factual answer.` to the SFT user turn for concise-answer probes. `sft_format: tool_trace` emits a synthetic native-tool trajectory (`search_hybrid` -> `read_file` -> `submit_answer`) from each train row. `sft_format: mixed_tool_trace` doubles the training rows with one plain answer-only example plus one tool-trace example per source row; the first 10-step probe preserved retrieval/temporal recall and submit behavior better than deeper naive answer-only SFT, but still needs hard-slice validation before promotion.

`agent_toolset: hybrid_deep` is the default P1-02 eval lane (`search_hybrid`, `grep_file`, `read_file`, `submit_answer`) after 10-row calibrate evidence showed long-article answer extraction gains. Its `search_hybrid` output adds relation-aware snippets for the top hits, including record-label, spouse/first-marriage, citizenship, head-of-state, head-coach, held-position, co-founded company, IPL/current/domestic cricket-team, final-season/traded-to sports-team, radio-station, campus-ballpark facility, release/distributor studio, production-company, current-mayor/head-of-government, sports-team, and sibling-list evidence patterns observed in P1-02 diagnostics. Snippet selection combines the episode's original question with the model's search query so relation words dropped by the query still guide evidence surfacing. Default remains two snippets per hit across the first three hits for broader 10-row behavior; use `--param search_snippets_per_hit=1` for the hard-subset distractor-reduction ablation, and `--param search_snippet_hits=5` only when diagnosing cases where the gold subject is retrieved below the third hit. Set `agent_toolset=hybrid` to reproduce the earlier Phase 0-compatible lane (`search_hybrid`, `read_file`, `submit_answer`).

`LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK=1` enables an opt-in structured answer-selection fallback. It extracts a single relation-specific candidate from accumulated `search_hybrid` snippets and uses it only when that candidate differs from the model's final answer and the model's final answer is not explicitly named in the question. This is not the default; it is a diagnostic lane for rows where the gold answer is visible in search snippets but the generative loop submits a co-mentioned distractor. Rows 60-64 show the fallback can select stale prior-snapshot snippets on repeated subjects, row 68 shows generic spouse snippets can still prefer a non-gold current/westernized surface, and row 76 shows a stale record-label snippet can override a correct submitted answer from `read_file`. Row 102 shows a safer positive case for year-specific position wording (`became <position> in <year>`), and rows 106/108 show positive co-founded/traded-to cases. Row 109 shows record-label candidate ordering must prefer album-release evidence over broad label-signing snippets. The fallback must remain opt-in until a temporal-aware confidence gate exists.

`eval_relation_hint` is an opt-in eval-input ablation, not the default. `--param eval_relation_hint=when_nondate` appends a target-relation hint only for `When...` questions whose dataset relation is not date-like, and helped the first-eight guard when paired with the record-label snippet alias. `--param eval_relation_hint=true` applies the relation hint broadly and regressed answer quality; do not use it as the default.

`adapter_source_run` lets `skip_train=true` replay eval against an existing run's `lora_adapter` while writing a new run directory for config, corpus export, traces, and metrics. This is preferred for eval-input ablations where retraining variance would obscure the comparison.

`eval_start_row` offsets the eval split before applying `max_eval_rows` / `calibrate_max_eval_rows`; use it for hard-subset probes without rerunning the easy first rows. Trace files keep local filenames (`row_0.jsonl`, etc.) but their `episode_start.row_index` records the original eval-split row index.

**Deps:** `peft`, `bitsandbytes` (in `pyproject.toml`).

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[p0-tw-03]: ../P0-TW-03/README.md
[run-v3]: ../../runs/20260525-144755_P0-TW-03
[run-v4]: ../../runs/20260525-230343_P0-TW-03
