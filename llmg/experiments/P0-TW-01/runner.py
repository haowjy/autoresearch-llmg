"""P0-TW-01 implementation — edit params in config.yaml for official runs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llmg.data.temporalwiki import dedupe_articles, load_tw_easy
from llmg.eval.rag.bm25 import BM25Index, recall_at_k

if TYPE_CHECKING:
    from llmg.observability.run_session import RunSession


def run(
    *,
    k: int = 5,
    eval_split: str = "test",
    session: RunSession | None = None,
) -> dict[str, float]:
    log = session.logger if session else logging.getLogger("llmg.run")

    log.info("loading dataset saxenan3/temporalwiki-drift-cl-easy")
    ds = load_tw_easy()
    corpus = dedupe_articles(ds["train"])
    log.info("index: %d articles from train split", len(corpus))

    index = BM25Index.from_corpus(corpus)
    split = ds[eval_split]
    queries = list(split["question"])
    gold = list(split["subject_sitelink"])
    log.info("eval: split=%s rows=%d top_k=%d", eval_split, len(queries), k)

    score = recall_at_k(index, queries, gold, k=k)
    log.info("retrieval_recall@%d = %.4f", k, score)

    metrics: dict[str, float] = {
        f"retrieval_recall@{k}": score,
        "index_docs": float(len(corpus)),
        "eval_rows": float(len(queries)),
    }
    if k == 5:
        metrics["retrieval_recall@5"] = score
    return metrics
