"""BM25 retrieval over article corpus."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class BM25Index:
    keys: list[str]
    bm25: BM25Okapi

    @classmethod
    def from_corpus(cls, corpus: dict[str, str]) -> BM25Index:
        keys = list(corpus.keys())
        tokenized = [tokenize(corpus[k]) for k in keys]
        return cls(keys=keys, bm25=BM25Okapi(tokenized))

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.keys[i] for i in ranked[:k]]


def recall_at_k(
    index: BM25Index,
    queries: list[str],
    gold_keys: list[str],
    k: int = 5,
) -> float:
    if not queries:
        return 0.0
    hits = 0
    for q, gold in zip(queries, gold_keys, strict=True):
        if gold in index.retrieve(q, k=k):
            hits += 1
    return hits / len(queries)
