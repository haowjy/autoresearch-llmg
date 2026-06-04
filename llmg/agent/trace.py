"""Structured JSONL traces for agent episodes (under llmg/runs/.../agent_traces/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Per-field caps for disk (tool loop may use a separate in-context cap).
TRACE_RAW_MAX = 8_000
TRACE_TOOL_MAX = 4_000
TRACE_SHELL_STDOUT_MAX = 4_000


def write_trace(path: Path | None, event: str, **fields: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"type": event, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def start_episode_trace(
    path: Path | None,
    *,
    question: str,
    as_of: str = "",
    model: str = "",
    toolset: str = "",
    gold_subject: str = "",
    gold_answer: str = "",
    row_index: int | None = None,
) -> None:
    """Overwrite path with episode header (one file per eval row)."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "type": "episode_start",
        "question": question,
        "as_of": as_of,
        "model": model,
        "toolset": toolset,
        "gold_subject": gold_subject,
        "gold_answer": gold_answer,
    }
    if row_index is not None:
        header["row_index"] = row_index
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")


def end_episode_trace(
    path: Path | None,
    *,
    answer: str,
    retrieved_doc_ids: list[str],
    steps: int,
    cmd_count: int,
    bytes_read: int,
    answer_source: str = "none",
    invalid_submit_count: int = 0,
) -> None:
    write_trace(
        path,
        "episode_end",
        answer=answer,
        retrieved_doc_ids=retrieved_doc_ids,
        steps=steps,
        cmd_count=cmd_count,
        bytes_read=bytes_read,
        answer_source=answer_source,
        invalid_submit_count=invalid_submit_count,
    )


def truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...(truncated)"
