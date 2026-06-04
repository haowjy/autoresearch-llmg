"""SFT example formatting for TemporalWiki Q/A (+ article context)."""

from __future__ import annotations

import re
from typing import Any, Literal

from datasets import Dataset

SftFormat = Literal[
    "plain",
    "concise",
    "tool_trace",
    "mixed_tool_trace",
]
CONCISE_ANSWER_INSTRUCTION = "Answer with only the short factual answer."
AGENTIC_SYSTEM = (
    "You are a research assistant with tools. The corpus is under articles/ (Markdown + YAML dates)."
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_")
    return slug or "Article"


def _evidence_window(article: str, answer: str, *, max_chars: int) -> str:
    text = " ".join(article.split())
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    answer_pos = text.lower().find(answer.lower()) if answer else -1
    if answer_pos < 0:
        return text[:max_chars]
    start = max(0, answer_pos - max_chars // 2)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _row_doc_id(row: dict[str, Any]) -> str:
    subject = str(row.get("subject_sitelink") or row.get("subject") or "Article")
    slice_tag = str(row.get("slice") or row.get("slice_tag") or "slice")
    return f"{_slug(subject)}__{_slug(slice_tag)}"


def _tool_trace_messages(
    row: dict[str, Any],
    *,
    max_article_chars: int,
) -> list[dict[str, Any]]:
    """Synthetic native-tool trajectory: search, read the supporting article, submit."""
    question = str(row.get("question") or "").strip()
    answer = str(row.get("object") or "").strip()
    subject = str(row.get("subject_sitelink") or row.get("subject") or "").strip()
    relation = str(row.get("relation") or "").strip()
    article = str(row.get("article") or "").strip()
    as_of = str(row.get("snapshot_new") or "").strip()
    doc_id = _row_doc_id(row)
    path = f"articles/{doc_id}.md"
    query = " ".join(part for part in (subject, relation) if part).strip() or question
    evidence = _evidence_window(article, answer, max_chars=max_article_chars)
    search_observation = f"1. {doc_id} score=1.0000 path={path}"
    if evidence:
        search_observation = f"{search_observation}\n   snippet: {evidence}"
    header = [
        "---",
        f"doc_id: {doc_id}",
        f"subject_sitelink: {subject}",
    ]
    if as_of:
        header.append(f"last_edited: '{as_of}'")
    header.append("---")
    read_observation = "\n".join(header)
    if evidence:
        read_observation = f"{read_observation}\n{evidence}"
    user = question if not as_of else f"{question}\n\n(as-of: {as_of})"
    return [
        {"role": "system", "content": AGENTIC_SYSTEM},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("search_hybrid", {"query": query})],
        },
        {"role": "tool", "name": "search_hybrid", "content": search_observation},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("read_file", {"path": path})],
        },
        {"role": "tool", "name": "read_file", "content": read_observation},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("submit_answer", {"answer": answer})],
        },
    ]


def row_to_messages(
    row: dict[str, Any],
    *,
    max_article_chars: int = 3000,
    sft_format: SftFormat = "plain",
) -> list[dict[str, Any]]:
    """User = question + article snippet; assistant = gold object (short fact)."""
    if sft_format in ("tool_trace", "mixed_tool_trace"):
        return _tool_trace_messages(row, max_article_chars=max_article_chars)
    question = str(row.get("question") or "").strip()
    article = str(row.get("article") or "").strip()
    if max_article_chars > 0 and article:
        article = article[:max_article_chars]
    if article:
        user = f"{question}\n\nArticle:\n{article}"
    else:
        user = question
    if sft_format == "concise":
        user = f"{user}\n\n{CONCISE_ANSWER_INSTRUCTION}"
    elif sft_format != "plain":
        raise ValueError(f"unknown sft_format: {sft_format}")
    answer = str(row.get("object") or "").strip()
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]


def build_sft_dataset(
    split: Dataset,
    *,
    max_rows: int | None = None,
    max_article_chars: int = 3000,
    sft_format: SftFormat = "plain",
) -> Dataset:
    n = len(split)
    if max_rows is not None:
        n = min(n, max_rows)
    rows = [split[i] for i in range(n)]
    if sft_format == "mixed_tool_trace":
        messages = []
        objects = []
        for r in rows:
            messages.append(
                row_to_messages(
                    r,
                    max_article_chars=max_article_chars,
                    sft_format="plain",
                )
            )
            objects.append(str(r.get("object") or ""))
            messages.append(
                row_to_messages(
                    r,
                    max_article_chars=max_article_chars,
                    sft_format="tool_trace",
                )
            )
            objects.append(str(r.get("object") or ""))
        return Dataset.from_dict({"messages": messages, "object": objects})
    return Dataset.from_dict(
        {
            "messages": [
                row_to_messages(
                    r,
                    max_article_chars=max_article_chars,
                    sft_format=sft_format,
                )
                for r in rows
            ],
            "object": [str(r.get("object") or "") for r in rows],
        }
    )
