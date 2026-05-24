"""Naive hybrid retrieval: BM25 (lexical) + bi-encoder dense + fusion.

Best-practice defaults for small article-level corpora (hundreds of docs):
- Run sparse and dense in parallel; fuse with RRF (rank-based, k=60) when scales differ.
- Alternative ``rerank``: BM25 candidate pool, then cosine rerank on candidates only (cheaper).
- Precompute document embeddings once per corpus (fine when N is modest).
- Wider first-stage pool than final k (``max(k * 4, 20)``) for recall.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal

from llmg.eval.rag.bm25 import BM25Index, tokenize

log = logging.getLogger(__name__)

FusionStrategy = Literal["rrf", "rerank"]

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RRF_K = 60
DEFAULT_STRATEGY: FusionStrategy = "rrf"
DEFAULT_CAND_MULT = 4
DEFAULT_MIN_CAND = 20
DEFAULT_RRF_POOL = 100


@dataclass(frozen=True)
class HybridHit:
    doc_id: str
    score: float
    rank: int


@lru_cache(maxsize=2)
def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@dataclass
class HybridIndex:
    """BM25 + dense index over one subject -> text corpus."""

    corpus: dict[str, str]
    bm25: BM25Index
    embed_model: str
    doc_embeddings: object | None  # np.ndarray (n_docs, dim), row-aligned with bm25.keys

    @classmethod
    def from_corpus(
        cls,
        corpus: dict[str, str],
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
        precompute_dense: bool = True,
    ) -> HybridIndex:
        if not corpus:
            return cls(corpus={}, bm25=BM25Index.from_corpus({}), embed_model=embed_model, doc_embeddings=None)
        bm25 = BM25Index.from_corpus(corpus)
        doc_embeddings = None
        if precompute_dense:
            try:
                import numpy as np

                model = _load_encoder(embed_model)
                texts = [corpus[k] for k in bm25.keys]
                doc_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
            except Exception as exc:
                log.warning("hybrid dense precompute failed (%s); BM25-only fallback", exc)
        return cls(
            corpus=corpus,
            bm25=bm25,
            embed_model=embed_model,
            doc_embeddings=doc_embeddings,
        )

    def _bm25_ranking(self, query: str, limit: int) -> list[str]:
        scores = self.bm25.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.bm25.keys[i] for i in ranked[:limit]]

    def _dense_ranking(self, query: str, limit: int) -> list[str]:
        if self.doc_embeddings is None or not self.bm25.keys:
            return []
        try:
            import numpy as np

            model = _load_encoder(self.embed_model)
            q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            q_emb = np.asarray(q_emb, dtype=np.float32)
            scores = (self.doc_embeddings @ q_emb.T).ravel()
            order = np.argsort(-scores)[:limit]
            return [self.bm25.keys[int(i)] for i in order]
        except Exception as exc:
            log.warning("hybrid dense ranking failed (%s)", exc)
            return []

    @staticmethod
    def _rrf_fuse(rankings: list[list[str]], *, rrf_k: int = DEFAULT_RRF_K) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, key in enumerate(ranking):
                fused[key] = fused.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: -x[1])

    def _retrieve_rrf(self, query: str, k: int, *, rrf_k: int) -> list[HybridHit]:
        pool = min(DEFAULT_RRF_POOL, max(k * 10, DEFAULT_MIN_CAND), len(self.bm25.keys))
        if pool <= 0:
            return []
        bm25_list = self._bm25_ranking(query, pool)
        dense_list = self._dense_ranking(query, pool)
        rankings = [bm25_list]
        if dense_list:
            rankings.append(dense_list)
        fused = self._rrf_fuse(rankings, rrf_k=rrf_k)[:k]
        return [HybridHit(doc_id=did, score=score, rank=i + 1) for i, (did, score) in enumerate(fused)]

    def _retrieve_rerank(self, query: str, k: int) -> list[HybridHit]:
        cand_k = min(max(k * DEFAULT_CAND_MULT, DEFAULT_MIN_CAND), len(self.bm25.keys))
        if cand_k <= 0:
            return []
        cands = self.bm25.retrieve(query, k=cand_k)
        if self.doc_embeddings is None:
            return [
                HybridHit(doc_id=did, score=float(cand_k - i), rank=i + 1)
                for i, did in enumerate(cands[:k])
            ]
        try:
            import numpy as np

            model = _load_encoder(self.embed_model)
            keys = cands
            texts = [self.corpus[key] for key in keys]
            q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            d_emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            q_emb = np.asarray(q_emb, dtype=np.float32)
            d_emb = np.asarray(d_emb, dtype=np.float32)
            scores = (d_emb @ q_emb.T).ravel()
            order = np.argsort(-scores)[:k]
            return [
                HybridHit(doc_id=keys[int(i)], score=float(scores[int(i)]), rank=r + 1)
                for r, i in enumerate(order)
            ]
        except Exception as exc:
            log.warning("hybrid rerank failed (%s); BM25 candidates", exc)
            return [
                HybridHit(doc_id=did, score=float(cand_k - i), rank=i + 1)
                for i, did in enumerate(cands[:k])
            ]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        strategy: FusionStrategy = DEFAULT_STRATEGY,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[HybridHit]:
        if not self.bm25.keys:
            return []
        if strategy == "rerank":
            return self._retrieve_rerank(query, k)
        return self._retrieve_rrf(query, k, rrf_k=rrf_k)

    def retrieve_doc_ids(
        self,
        query: str,
        k: int = 5,
        **kwargs: object,
    ) -> list[str]:
        return [h.doc_id for h in self.retrieve(query, k, **kwargs)]  # type: ignore[arg-type]


_index_cache: dict[tuple[str, str, str], HybridIndex] = {}


def get_hybrid_index(
    corpus: dict[str, str],
    *,
    cache_key: str | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    strategy: FusionStrategy = DEFAULT_STRATEGY,
) -> HybridIndex:
    """Return a cached HybridIndex for a corpus (agent loop: one build per export dir)."""
    key = (cache_key or "default", embed_model, strategy)
    if key not in _index_cache:
        _index_cache[key] = HybridIndex.from_corpus(corpus, embed_model=embed_model)
    return _index_cache[key]


def clear_hybrid_index_cache() -> None:
    _index_cache.clear()


def format_hits_for_agent(
    hits: list[HybridHit],
    *,
    path_for_doc_id: Callable[[str], str | Path | None] | None = None,
) -> str:
    """Human-readable lines for the LLM tool observation."""
    if not hits:
        return "no hits"
    lines: list[str] = []
    for h in hits:
        extra = ""
        if path_for_doc_id is not None:
            p = path_for_doc_id(h.doc_id)
            if p is not None:
                extra = f" path={p}"
        lines.append(f"{h.rank}. {h.doc_id} score={h.score:.4f}{extra}")
    return "\n".join(lines)


def hybrid_index_from_fs(corpus_root: Path, **kwargs: object) -> HybridIndex:
    from llmg.memory.fs_store import FsStore

    store = FsStore(corpus_root)
    corpus = store.corpus_dict()
    cache_key = str(corpus_root.resolve())
    return get_hybrid_index(corpus, cache_key=cache_key, **kwargs)  # type: ignore[arg-type]
