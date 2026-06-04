"""Retrieval and answer metrics for time-sliced TemporalWiki eval."""

from __future__ import annotations

import logging
from datetime import date
from typing import Sequence

log = logging.getLogger(__name__)

# Cosine >= threshold counts as a "hit" (answer_cosine_hit_rate).
ANSWER_COSINE_HIT_THRESHOLD = 0.85

_encoder_error_logged = False


def parse_iso_date(value: str) -> date | None:
    value = (value or "").strip()[:10]
    if len(value) < 10:
        return None
    try:
        y, m, d = int(value[:4]), int(value[5:7]), int(value[8:10])
        return date(y, m, d)
    except (ValueError, TypeError):
        return None


def as_of_matches_slice_end(
    as_of: str,
    first_edited: str,
    last_edited: str,
) -> bool:
    """True when question as-of equals the article's ``last_edited`` (slice end / snapshot_new).

    ``first_edited`` is metadata only for now; eval rows use ``snapshot_new`` as as-of.
    """
    _ = first_edited
    q = parse_iso_date(as_of)
    end = parse_iso_date(last_edited)
    if q is None or end is None:
        return False
    return q == end


# Backward-compatible alias
as_of_in_doc_window = as_of_matches_slice_end


def subject_recall_at_k(
    retrieved_subjects: Sequence[str],
    gold_subject: str,
    k: int,
) -> bool:
    return gold_subject in list(retrieved_subjects)[:k]


def temporal_recall_at_k(
    retrieved_doc_ids: Sequence[str],
    *,
    gold_subject: str,
    as_of: str,
    doc_meta: dict[str, tuple[str, str, str]],  # doc_id -> (subject, first_edited, last_edited)
    k: int,
) -> bool:
    """Hit if some top-k doc matches subject and as-of equals that doc's ``last_edited``."""
    for doc_id in list(retrieved_doc_ids)[:k]:
        meta = doc_meta.get(doc_id)
        if meta is None:
            continue
        subj, first_ed, last_ed = meta
        if subj != gold_subject:
            continue
        if as_of_matches_slice_end(as_of, first_ed, last_ed):
            return True
    return False


def answer_exact_match(pred: str, gold: str) -> float:
    import re

    def norm(t: str) -> str:
        t = t.strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^\w\s]", "", t)
        return t

    if not pred:
        return 0.0
    return 1.0 if norm(pred) == norm(gold) else 0.0


def _answer_encoder():
    from llmg.search.embeddings import DEFAULT_EMBED_MODEL, get_sentence_embedder

    return get_sentence_embedder(DEFAULT_EMBED_MODEL)


def clear_answer_encoder_cache() -> None:
    global _encoder_error_logged
    from llmg.search.embeddings import clear_sentence_embedder_cache

    clear_sentence_embedder_cache()
    _encoder_error_logged = False


def answer_cosine_similarity(pred: str, gold: str) -> float:
    scores = answer_cosine_similarity_batch([(pred, gold)])
    return scores[0] if scores else 0.0


def answer_cosine_similarity_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Cosine similarity per (pred, gold) pair; empty strings score 0."""
    global _encoder_error_logged
    if not pairs:
        return []
    texts: list[str] = []
    out: list[float] = [0.0] * len(pairs)
    index_map: list[int] = []
    for i, (pred, gold) in enumerate(pairs):
        if not pred or not gold:
            continue
        texts.append(pred)
        texts.append(gold)
        index_map.append(i)
    if not texts:
        return out
    try:
        import numpy as np

        model = _answer_encoder()
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        for j, row_i in enumerate(index_map):
            pred_emb = emb[j * 2]
            gold_emb = emb[j * 2 + 1]
            out[row_i] = float(pred_emb @ gold_emb)
        return out
    except Exception:
        if not _encoder_error_logged:
            log.exception("answer_cosine_similarity_batch failed")
            _encoder_error_logged = True
        raise


def answer_cosine_hit_rate(scores: Sequence[float], *, threshold: float = ANSWER_COSINE_HIT_THRESHOLD) -> float:
    if not scores:
        return 0.0
    hits = sum(1 for s in scores if s >= threshold)
    return hits / len(scores)
