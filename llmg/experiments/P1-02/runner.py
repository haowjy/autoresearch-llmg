"""P1-02 — QLoRA + RAG (scaffold)."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def run(session=None, **params: Any) -> dict[str, float]:
    """Calibrate/offical entry; training not implemented yet."""
    _ = session
    log.info("P1-02 params=%s", params)
    raise NotImplementedError(
        "P1-02 QLoRA training + RAG eval is not implemented. "
        "See llmg/experiments/P1-02/README.md for the checklist."
    )
