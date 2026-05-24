"""Tool definitions for native function-calling agents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Literal

from llmg.search.hybrid import format_hits_for_agent, hybrid_index_from_fs

from llmg.agent.sandbox import AgentSandbox
from llmg.agent.state import AgentEpisodeState
from llmg.memory.fs_store import FsStore

ToolFn = Callable[..., str]
AgentToolset = Literal["shell", "hybrid", "full"]

# Bounds aligned with TemporalWiki `object` gold (short phrase, not prose).
SUBMIT_ANSWER_MAX_WORDS = 15
SUBMIT_ANSWER_MAX_CHARS = 120


def normalize_submitted_answer(raw: str) -> str:
    """First line, stripped — the string we score with answer_em."""
    text = raw.strip().split("\n")[0].strip()
    # Drop wrapping quotes the model sometimes adds.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def validate_submitted_answer(text: str) -> str | None:
    """Return an error message for the model, or None if acceptable."""
    if not text:
        return "empty answer"
    words = text.split()
    if len(words) > SUBMIT_ANSWER_MAX_WORDS:
        return (
            f"too long ({len(words)} words; max {SUBMIT_ANSWER_MAX_WORDS}). "
            "Submit only the minimal fact (name, date, title, number)."
        )
    if len(text) > SUBMIT_ANSWER_MAX_CHARS:
        return f"too long ({len(text)} chars; max {SUBMIT_ANSWER_MAX_CHARS})."
    lower = text.lower()
    rejected_prefixes = (
        "the article",
        "the provided",
        "based on",
        "according to",
        "i cannot",
        "there is no",
        "it is not",
        "the document",
    )
    if any(lower.startswith(p) for p in rejected_prefixes):
        return "looks like explanation, not a fact — submit only the answer (e.g. 'Project Cadmus')."
    return None


def supports_native_tools(tokenizer) -> bool:
    """True when tokenizer chat template encodes tools (e.g. Gemma 4)."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return "<tool" in template.lower() or "<|tool" in template.lower()


def execute_parsed_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    sandbox: AgentSandbox,
    fs_store: FsStore,
    retrieved_doc_ids: list[str],
    tools_by_name: dict[str, ToolFn],
    episode: AgentEpisodeState,
) -> list[dict[str, str]]:
    """Run parsed tool_calls; return tool role messages for chat history."""
    messages: list[dict[str, str]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = str(fn.get("name", ""))
        arguments = fn.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        tool_fn = tools_by_name.get(name)
        if tool_fn is None:
            content = f"error: unknown tool {name!r}"
        else:
            try:
                content = tool_fn(**arguments)
            except TypeError as exc:
                content = f"error: bad arguments for {name}: {exc}"
            except Exception as exc:
                content = f"error: {exc!r}"

        if name != "submit_answer":
            _record_retrieval_from_tool(name, arguments, content, fs_store, retrieved_doc_ids, sandbox)
        messages.append({"role": "tool", "content": content, "name": name})
    return messages


def _record_retrieval_from_tool(
    name: str,
    arguments: dict[str, Any],
    content: str,
    fs_store: FsStore,
    retrieved_doc_ids: list[str],
    sandbox: AgentSandbox,
) -> None:
    def _add(doc_id: str | None) -> None:
        if doc_id and doc_id not in retrieved_doc_ids:
            retrieved_doc_ids.append(doc_id)

    if name == "read_file":
        rel = str(arguments.get("path", ""))
        path = (sandbox.workspace / rel).resolve()
        _add(fs_store.doc_id_from_path(path))
    elif name == "search_hybrid":
        for line in content.splitlines():
            line = line.strip()
            if not line or line == "no hits":
                continue
            path_m = re.search(r"\bpath=(\S+)", line)
            if path_m:
                _add(fs_store.doc_id_from_path(sandbox.workspace / path_m.group(1)))
    elif name == "run_shell":
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("exit_code"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = sandbox.workspace / line
            if path.suffix == ".md" or str(line).endswith(".md"):
                _add(fs_store.doc_id_from_path(path))


def make_tool_functions(
    *,
    sandbox: AgentSandbox,
    fs_store: FsStore,
    episode: AgentEpisodeState,
    toolset: AgentToolset = "shell",
) -> tuple[list[ToolFn], dict[str, ToolFn]]:
    """Build Python tools for tokenizer.apply_chat_template(tools=...)."""
    hybrid_index = (
        hybrid_index_from_fs(sandbox.workspace) if toolset in ("hybrid", "full") else None
    )

    def run_shell(command: str) -> str:
        """Run one allowlisted shell command in the corpus workspace.

        Args:
            command: Single command without pipes or chaining. Prefer `rg -i 'term' articles/`
                or `grep -ri term articles/`. Binaries: rg, grep, find, head, cat, ls, sqlite3.

        Returns:
            stdout and stderr from the command (truncated).
        """
        try:
            result = sandbox.run(command)
        except Exception as exc:
            return f"error: {exc!r}"
        parts = [f"exit_code={result.returncode}"]
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        return "\n".join(parts)

    def read_file(path: str) -> str:
        """Read a markdown article under the corpus workspace.

        Args:
            path: Relative path from search output, e.g. `articles/Conner_Kent.md`.

        Returns:
            File body and YAML frontmatter (truncated).
        """
        file_path = (sandbox.workspace / path).resolve()
        if not str(file_path).startswith(str(sandbox.workspace.resolve())):
            return "error: path escapes workspace"
        if not file_path.is_file():
            return f"error: not found: {path}"
        return file_path.read_text(encoding="utf-8")[:4000]

    def search_hybrid(query: str, k: int = 5) -> str:
        """Search the corpus with BM25 + dense vectors (RRF fusion).

        Args:
            query: Natural-language question or keywords.
            k: Number of article hits to return (default 5).

        Returns:
            Ranked subjects with scores and relative paths under articles/.
        """
        if hybrid_index is None:
            return "error: hybrid index not available"
        hits = hybrid_index.retrieve(query, k=k)

        def _rel_path(doc_id: str) -> str | None:
            p = fs_store.path_for_doc_id(doc_id)
            if p is None:
                return None
            try:
                return str(p.relative_to(sandbox.workspace))
            except ValueError:
                return str(p)

        return format_hits_for_agent(hits, path_for_doc_id=_rel_path)

    def submit_answer(answer: str) -> str:
        """Finish the episode — call once you know the fact that answers the question.

        Args:
            answer: The minimal factual answer only: entity name, date, job title, number,
                or short phrase (examples: "Project Cadmus", "CNN", "1964").
                No full sentences, apologies, or "the article says…".

        Returns:
            Confirmation, or an error asking you to shorten or fix the answer.
        """
        text = normalize_submitted_answer(answer)
        err = validate_submitted_answer(text)
        if err:
            return f"error: {err} Try submit_answer again with just the fact."
        episode.final_answer = text
        episode.done = True
        return "Answer recorded. Stop calling tools."

    tools: list[ToolFn] = []
    if toolset in ("hybrid", "full"):
        tools.append(search_hybrid)
    if toolset in ("shell", "full"):
        tools.append(run_shell)
    tools.extend([read_file, submit_answer])
    by_name = {t.__name__: t for t in tools}
    return tools, by_name
