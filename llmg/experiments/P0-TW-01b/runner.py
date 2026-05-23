"""P0-TW-01b — retention retrieval with train+stable index."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llmg.data.temporalwiki import load_tw_easy, merge_corpus_splits
from llmg.eval.rag.bm25 import BM25Index, recall_at_k

if TYPE_CHECKING:
    from llmg.observability.run_session import RunSession


def run(
    *,
    k: int = 5,
    index_splits: list[str] | None = None,
    eval_split: str = "stable",
    report_test_on_same_index: bool = True,
    session: RunSession | None = None,
) -> dict[str, float]:
    log = session.logger if session else logging.getLogger("llmg.run")
    splits = index_splits if index_splits is not None else ["train", "stable"]

    log.info("loading dataset saxenan3/temporalwiki-drift-cl-easy")
    ds = load_tw_easy()
    corpus = merge_corpus_splits(ds, splits)
    log.info("index: %d articles from splits %s", len(corpus), splits)

    index = BM25Index.from_corpus(corpus)
    eval_rows = ds[eval_split]
    queries = list(eval_rows["question"])
    gold = list(eval_rows["subject_sitelink"])
    log.info("eval: split=%s rows=%d top_k=%d", eval_split, len(queries), k)

    score = recall_at_k(index, queries, gold, k=k)
    log.info("retrieval_recall@%d (%s) = %.4f", k, eval_split, score)

    metrics: dict[str, float] = {
        f"retrieval_recall@{k}": score,
        "index_docs": float(len(corpus)),
        "eval_rows": float(len(queries)),
    }
    if k == 5:
        metrics["retrieval_recall@5"] = score

    if report_test_on_same_index and eval_split != "test":
        test_q = list(ds["test"]["question"])
        test_g = list(ds["test"]["subject_sitelink"])
        test_score = recall_at_k(index, test_q, test_g, k=k)
        metrics[f"retrieval_recall@{k}_test_same_index"] = test_score
        log.info(
            "retrieval_recall@%d (test, same index) = %.4f",
            k,
            test_score,
        )

    return metrics
