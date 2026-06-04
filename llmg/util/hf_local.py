"""Hugging Face Hub offline / local-only mode for repeat experiment runs."""

from __future__ import annotations

import os

_ENV_FLAG = "LLMG_HF_LOCAL_ONLY"


def hf_local_files_only() -> bool:
    """True when ``LLMG_HF_LOCAL_ONLY`` is set (1, true, yes)."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "yes")


def configure_hf_offline_if_requested() -> None:
    """Set standard HF offline env vars when ``LLMG_HF_LOCAL_ONLY`` is enabled."""
    if not hf_local_files_only():
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
