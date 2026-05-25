# Datasets — temporal memory & continual learning

Catalog for the LLMG program. **In harness** = `llmg.run` or download step exists today. **Planned** = charter / roadmap candidates.

Link style: narrative `[label][ref-id]`; full URLs only under [References](#references).

---

## In harness (active)

| Dataset | Role | HF / code |
|---------|------|-----------|
| **TemporalWiki drift CL (easy)** | Phase 0 RAG floor; acquisition (`test`) + retention (`stable`) | [tw-easy][tw-easy] |
| **TemporalWiki drift CL (base)** | Parent variant (string-concat prompts); same snapshot backbone | [tw-cl][tw-cl] |

```bash
hf download saxenan3/temporalwiki-drift-cl-easy --repo-type dataset
hf download saxenan3/temporalwiki-drift-cl --repo-type dataset
uv run python -m llmg.run --experiment P0-TW-03 --run-phase calibrate   # harness matrix (v2, easy)
uv run python -m llmg.run --experiment P0-TW-03 --run-phase official  # pinned agent cells (easy)
uv run python -m llmg.run --experiment P0-TW-04 --run-phase calibrate   # harder CL base (pinned harness)
uv run python -m llmg.run --experiment P0-TW-04 --run-phase official  # pinned harness + agent (base)
# Deprecated: P0-TW-01 / P0-TW-01b (collapsed index; archaeology only)
```

**Snapshots (easy variant):** `2025-11-20` → `2025-12-01` → `2026-01-01` → `2026-02-01` (3 drift slices + stable probe). See [temporal-datasets wiki][wiki-temporal] in KB.

---

## Temporal / continual QA (planned eval & training)

| Dataset | Granularity | Fit for LLMG |
|---------|-------------|--------------|
| **TemporalWiki** (EMNLP 2022) | 5 snapshots (2021.08–12); diff + probes | Original lifelong wiki benchmark; [paper][tw-2022-paper] · [HF][tw-2022-hf] · [code][tw-2022-code] |
| **StreamingQA** | 14y news; `question_ts` / `evidence_ts`; quarterly eval | Fine-grained horizon vs 4-month TW; [paper][streamingqa-paper] · [code][streamingqa-code] |
| **PAT-Questions** | Present-anchored; self-updating via Wikidata SPARQL | Stale-trap / “answer for today”; [repo][pat-repo] · [paper][pat-paper] |
| **RealTimeQA** | Evolving world knowledge (NAACL 2024) | Compares TW, StreamingQA, etc.; [paper][realtimeqa-paper] |
| **Time-Sensitive QA** | Passage facts evolve in text | Template + annotated splits; [code][tsqa-code] |
| **QA under temporal conflict** | Doc streams → structured (S,R,O,t) KB | Supersession-aware RAG; [paper][temporal-conflict-paper] |
| **Diachronic RAG** | Chunk time intervals + as-of retrieval | Engineering pattern for timeline queries; [paper][diachronic-rag-paper] |
| **CLTSQA** | Continual temporal-sensitive QA training | Extends TSQA with CL splits; [paper][cltsqa-paper] |

---

## Narrative / book (planned — dense timelines)

| Dataset | Role | Link |
|---------|------|------|
| **ChronoQA** | 18 books; chronology + causal RAG | [chronoqa][chronoqa-hf] |
| **NarrativeXL** | Long-horizon memory; answer depends on read progress | [paper][narrativexl-paper] |
| **Torque** | Temporal ordering in news MRC | [paper][torque-paper] |

Good for **chapter = day** synthetic curricula and stale-trap eval (retrieve early vs late passage).

---

## Machine unlearning (reference only — not primary LLMG metric)

LLMG targets **graceful supersession** (drop obsolete facts on *current* questions, retain *stable* facts). That is **not** the same as TOFU/MUSE **forget sets**. See [KB decision 0006][dec-0006].

| Benchmark | Link |
|-----------|------|
| **OpenUnlearning** (TOFU, MUSE, WMDP) | [open-unlearning][open-unlearning] |
| **TOFU** | [tofu-hf][tofu-hf] |
| **MUSE** (news + books) | via OpenUnlearning |

---

## Infrastructure (versioned KB, not a QA dataset)

| Tool | Role | Link |
|------|------|------|
| **Graphiti** | Temporal knowledge graph; `valid_at` / `expired_at` edges | [graphiti][graphiti] |

---

## Custom corpora (program-defined)

| ID | Source | Status |
|----|--------|--------|
| **P-1-REPO** | Git history + Meridian work/KB (`autoresearch-llmg`, research-docs) | Planned (charter Phase -1) |
| **P0-SYN** | Synthetic org/wiki with daily supersede + as-of labels | Planned (charter Phase 4) |
| **P0-BOOK** | Public-domain book + injected edits | Planned |

---

## References

[tw-easy]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl-easy
[tw-cl]: https://huggingface.co/datasets/saxenan3/temporalwiki-drift-cl
[tw-2022-hf]: https://huggingface.co/datasets/seonghyeonye/TemporalWiki
[tw-2022-code]: https://github.com/joeljang/temporalwiki
[tw-2022-paper]: https://aclanthology.org/2022.emnlp-main.418/
[streamingqa-paper]: https://arxiv.org/abs/2205.11388
[streamingqa-code]: https://github.com/google-deepmind/streamingqa
[pat-repo]: https://github.com/jannatmeem95/PAT-Questions
[pat-paper]: https://aclanthology.org/2024.findings-acl.777/
[realtimeqa-paper]: https://aclanthology.org/2024.naacl-long.302/
[tsqa-code]: https://github.com/wenhuchen/time-sensitive-qa
[temporal-conflict-paper]: https://arxiv.org/abs/2506.07270
[diachronic-rag-paper]: https://arxiv.org/abs/2507.22917
[cltsqa-paper]: https://arxiv.org/abs/2407.12470
[chronoqa-hf]: https://huggingface.co/datasets/zy113/ChronoQA
[narrativexl-paper]: https://aclanthology.org/2023.findings-emnlp.1005/
[torque-paper]: https://arxiv.org/abs/2005.00242
[open-unlearning]: https://github.com/locuslab/open-unlearning
[tofu-hf]: https://huggingface.co/datasets/locuslab/TOFU
[graphiti]: https://github.com/getzep/graphiti
[wiki-temporal]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/temporal-datasets.md
[dec-0006]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/decisions/0006-graceful-supersession-not-unlearning.md
