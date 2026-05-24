# LLMG roadmap (short index)

Full experiment rows: [experiment-registry.md][wiki-registry] (`llmg/kb/wiki/experiment-registry.md`).

Program log: [RESEARCH-LOG.md][research-log]

| Phase | Campaign slug | Gate | First experiment |
|-------|---------------|------|------------------|
| 0 | `phase0-temporalwiki-rag` | 0 (RAG ceiling) | **P0-TW-03** done (Gate 0 passed); 01/01b deprecated; **P1-02** next |
| 1 | `phase1-protocol-lora` | 1 | P1-02 protocol LoRA + RAG |
| 2 | `phase2-stacked-lora` | 2 | P2 grid stacked vs rank-matched |
| -1 | `phase-minus1-data-inventory` | — | P-1-01 corpus audit |

**Hardware default:** RTX 3090 24GB, `google/gemma-4-E4B-it`, text-only, train 4k + long mix 8k/16k.

**Datasets:** [DATASETS.md][datasets] (active + planned corpora with links).

---

## References

[wiki-registry]: https://github.com/haowjy/research-docs/blob/main/llmg/kb/wiki/experiment-registry.md
[datasets]: DATASETS.md
[research-log]: ../RESEARCH-LOG.md
