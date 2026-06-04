"""Shared sentence-transformer loader (hybrid retrieval + answer cosine metrics)."""

from __future__ import annotations

from functools import lru_cache

from llmg.util.hf_local import hf_local_files_only

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_sentence_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """Load a cached ``SentenceTransformer`` (one Hub resolution per model name)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        local_files_only=hf_local_files_only(),
    )


def clear_sentence_embedder_cache() -> None:
    get_sentence_embedder.cache_clear()
