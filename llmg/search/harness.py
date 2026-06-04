"""Unified retrieval harness: BM25, hybrid dense, ripgrep on filesystem."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llmg.data.corpus_export import records_to_corpus
from llmg.eval.rag.bm25 import BM25Index
from llmg.eval.temporal_metrics import (
    subject_recall_at_k,
    temporal_recall_at_k,
)
from llmg.memory.doc_catalog import DocCatalog
from llmg.memory.fs_store import FsStore
from llmg.memory.sqlite_store import SqliteStore
from llmg.search.hybrid import HybridIndex

log = logging.getLogger(__name__)

DataFormat = Literal["memory", "sqlite", "filesystem"]
SearchMode = Literal["harness_bm25", "harness_hybrid", "harness_rg"]


@dataclass
class HarnessResult:
    retrieval_recall_at_k: float
    temporal_recall_at_k: float
    elapsed_s: float


def _load_corpus(
    data_format: DataFormat,
    *,
    memory_corpus: dict[str, str] | None,
    fs_store: FsStore | None,
    sql_store: SqliteStore | None,
) -> dict[str, str]:
    if data_format == "memory":
        if not memory_corpus:
            raise ValueError("memory_corpus required for data_format=memory")
        return memory_corpus
    if data_format == "filesystem":
        if fs_store is None:
            raise ValueError("fs_store required for data_format=filesystem")
        return fs_store.corpus_dict()
    if data_format == "sqlite":
        if sql_store is None:
            raise ValueError("sql_store required for data_format=sqlite")
        return sql_store.corpus_dict()
    raise ValueError(f"unknown data_format: {data_format!r}")


_bm25_index_cache: dict[int, BM25Index] = {}
_hybrid_index_cache: dict[int, HybridIndex] = {}


def get_bm25_index(corpus: dict[str, str]) -> BM25Index:
    cache_id = id(corpus)
    index = _bm25_index_cache.get(cache_id)
    if index is None:
        index = BM25Index.from_corpus(corpus)
        _bm25_index_cache[cache_id] = index
    return index


def retrieve_bm25(corpus: dict[str, str], query: str, k: int) -> list[str]:
    return get_bm25_index(corpus).retrieve(query, k=k)


def clear_harness_caches() -> None:
    _bm25_index_cache.clear()
    _hybrid_index_cache.clear()
    from llmg.search.embeddings import clear_sentence_embedder_cache
    from llmg.search.hybrid import clear_hybrid_index_cache

    clear_hybrid_index_cache()
    clear_sentence_embedder_cache()


def retrieve_hybrid(
    corpus: dict[str, str],
    query: str,
    k: int,
    *,
    strategy: str = "rrf",
) -> list[str]:
    cache_id = id(corpus)
    hybrid = _hybrid_index_cache.get(cache_id)
    if hybrid is None:
        hybrid = HybridIndex.from_corpus(corpus)
        _hybrid_index_cache[cache_id] = hybrid
    return hybrid.retrieve_doc_ids(query, k, strategy=strategy)  # type: ignore[arg-type]


def _python_grep_articles(fs_store: FsStore, terms: list[str], limit: int) -> list[Path]:
    hits: list[Path] = []
    for path in sorted(fs_store.articles_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8").lower()
        except OSError:
            continue
        if any(t in text for t in terms):
            hits.append(path)
        if len(hits) >= limit:
            break
    return hits


def retrieve_rg(fs_store: FsStore, query: str, k: int) -> list[str]:
    """Return doc_ids for articles matching a grep/rg query."""
    articles_dir = fs_store.articles_dir
    if not articles_dir.is_dir():
        return []
    stop = frozenset(
        {"who", "what", "when", "where", "which", "the", "and", "for", "as", "of", "is", "are", "was", "were"}
    )
    terms = [
        t
        for t in re.findall(r"[a-z][a-z0-9]{2,}", query.lower())
        if t not in stop and not re.fullmatch(r"20\d{2}", t)
    ][:5]
    if not terms:
        terms = re.findall(r"[a-z0-9]{3,}", query.lower())[:3]
    if not terms:
        return []
    pattern = "|".join(re.escape(t) for t in terms)
    paths: list[Path] = []
    try:
        proc = subprocess.run(
            ["rg", "-l", "-i", pattern, str(articles_dir)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        paths = [Path(p.strip()) for p in proc.stdout.splitlines() if p.strip()]
    except FileNotFoundError:
        try:
            proc = subprocess.run(
                ["grep", "-ril", "-E", pattern, str(articles_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            paths = [Path(p.strip()) for p in proc.stdout.splitlines() if p.strip()]
        except FileNotFoundError:
            paths = _python_grep_articles(fs_store, terms, limit=k * 3)
    doc_ids: list[str] = []
    seen: set[str] = set()
    for path in paths:
        did = fs_store.doc_id_from_path(path)
        if did and did not in seen:
            seen.add(did)
            doc_ids.append(did)
        if len(doc_ids) >= k:
            break
    return doc_ids[:k]


def recall_at_k_harness(
    *,
    search_mode: SearchMode,
    data_format: DataFormat,
    queries: list[str],
    gold_subjects: list[str],
    as_of_dates: list[str],
    k: int,
    doc_meta: dict[str, tuple[str, str, str]],
    memory_corpus: dict[str, str] | None = None,
    fs_store: FsStore | None = None,
    sql_store: SqliteStore | None = None,
) -> tuple[float, float]:
    if not queries:
        return 0.0, 0.0

    corpus = _load_corpus(
        data_format,
        memory_corpus=memory_corpus,
        fs_store=fs_store,
        sql_store=sql_store,
    )
    subject_hits = 0
    temporal_hits = 0
    for i, (q, gold) in enumerate(zip(queries, gold_subjects, strict=True)):
        as_of = as_of_dates[i] if i < len(as_of_dates) else ""
        if search_mode == "harness_hybrid":
            retrieved = retrieve_hybrid(corpus, q, k)
        elif search_mode == "harness_rg":
            if fs_store is None:
                raise ValueError("harness_rg requires fs_store")
            retrieved = retrieve_rg(fs_store, q, k)
        else:
            retrieved = retrieve_bm25(corpus, q, k)
        if subject_recall_at_k(
            [doc_meta[d][0] for d in retrieved if d in doc_meta],
            gold,
            k,
        ):
            subject_hits += 1
        if temporal_recall_at_k(retrieved, gold_subject=gold, as_of=as_of, doc_meta=doc_meta, k=k):
            temporal_hits += 1
    n = len(queries)
    return subject_hits / n, temporal_hits / n


def run_harness_cell(
    *,
    search_mode: SearchMode,
    data_format: DataFormat,
    queries: list[str],
    gold_subjects: list[str],
    as_of_dates: list[str],
    k: int,
    catalog: DocCatalog | None = None,
    memory_corpus: dict[str, str] | None = None,
    corpus_root: Path | None = None,
) -> HarnessResult:
    import time

    fs_store = FsStore(corpus_root) if corpus_root else None
    sql_store = (
        SqliteStore(corpus_root / "corpus.db") if corpus_root and data_format == "sqlite" else None
    )
    if data_format == "memory" and memory_corpus is None:
        raise ValueError("memory_corpus required")
    if catalog is None:
        if corpus_root:
            from llmg.data.corpus_export import require_versioned_corpus

            require_versioned_corpus(corpus_root)
            catalog = DocCatalog.from_corpus_root(corpus_root)
        else:
            raise ValueError("catalog or corpus_root required")

    doc_meta = catalog.doc_meta_tuple()
    t0 = time.monotonic()
    sub_r, temp_r = recall_at_k_harness(
        search_mode=search_mode,
        data_format=data_format,
        queries=queries,
        gold_subjects=gold_subjects,
        as_of_dates=as_of_dates,
        k=k,
        doc_meta=doc_meta,
        memory_corpus=memory_corpus,
        fs_store=fs_store,
        sql_store=sql_store,
    )
    return HarnessResult(
        retrieval_recall_at_k=sub_r,
        temporal_recall_at_k=temp_r,
        elapsed_s=time.monotonic() - t0,
    )
