"""Tool definitions for native function-calling agents.

Harness note: `agent_term_hybrid` keeps read_file default (first 4000 chars, no grep_file).
`agent_term_hybrid_deep` adds grep_file and documents offset/limit pagination for long articles.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable, Literal

from llmg.search.hybrid import format_hits_for_agent, hybrid_index_from_fs

from llmg.agent.sandbox import AgentSandbox
from llmg.agent.state import AgentEpisodeState
from llmg.memory.fs_store import FsStore

ToolFn = Callable[..., str]
AgentToolset = Literal["shell", "hybrid", "hybrid_deep", "full"]

READ_FILE_DEFAULT_LIMIT = 4000
READ_FILE_MAX_LIMIT = 8000
GREP_FILE_OUTPUT_MAX = 2000
GREP_FILE_MAX_MATCHES = 50

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

    if name in ("read_file", "grep_file"):
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


def _resolve_workspace_file(sandbox: AgentSandbox, path: str) -> Path | str:
    """Return resolved file path, or an error string."""
    rel = path.strip().lstrip("/")
    if not rel:
        return "error: empty path"
    file_path = (sandbox.workspace / rel).resolve()
    if not str(file_path).startswith(str(sandbox.workspace.resolve())):
        return "error: path escapes workspace"
    return file_path


def make_tool_functions(
    *,
    sandbox: AgentSandbox,
    fs_store: FsStore,
    episode: AgentEpisodeState,
    toolset: AgentToolset = "shell",
) -> tuple[list[ToolFn], dict[str, ToolFn]]:
    """Build Python tools for tokenizer.apply_chat_template(tools=...)."""
    hybrid_index = (
        hybrid_index_from_fs(sandbox.workspace)
        if toolset in ("hybrid", "hybrid_deep", "full")
        else None
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

    def read_file(path: str, offset: int = 0, limit: int = READ_FILE_DEFAULT_LIMIT) -> str:
        """Read a slice of a markdown article under the corpus workspace.

        After search_hybrid, open the hit path here. Use offset/limit to paginate long
        articles when the fact is not in the first slice; or use grep_file (hybrid_deep).

        Args:
            path: Relative path from search output, e.g. `articles/Conner_Kent__Nov_Dec.md`.
            offset: Character offset into the file (default 0 = start).
            limit: Max characters to return (default 4000).

        Returns:
            File slice from offset (truncated to limit).
        """
        resolved = _resolve_workspace_file(sandbox, path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.is_file():
            return f"error: not found: {path}"
        off = max(0, int(offset))
        lim = max(1, min(int(limit), READ_FILE_MAX_LIMIT))
        text = resolved.read_text(encoding="utf-8")
        if off >= len(text):
            return f"(empty slice: offset {off} past end {len(text)})"
        return text[off : off + lim]

    def grep_file(path: str, pattern: str, max_matches: int = 20) -> str:
        """Search one article file for a pattern (ripgrep or grep, no pipes).

        Use on a path from search_hybrid when read_file slices miss the fact.

        Args:
            path: Relative path under the workspace, e.g. `articles/Subject__slice.md`.
            pattern: Single-line regex or literal (no newlines).
            max_matches: Cap on matching lines (default 20).

        Returns:
            Matching lines with line numbers (truncated).
        """
        resolved = _resolve_workspace_file(sandbox, path)
        if isinstance(resolved, str):
            return resolved
        if not resolved.is_file():
            return f"error: not found: {path}"
        if not pattern or "\n" in pattern or "\r" in pattern:
            return "error: pattern must be a non-empty single line"
        n = max(1, min(int(max_matches), GREP_FILE_MAX_MATCHES))
        rel = str(resolved.relative_to(sandbox.workspace))
        pat = shlex.quote(pattern)
        rel_q = shlex.quote(rel)
        for cmd in (
            f"rg -n -i -m {n} {pat} {rel_q}",
            f"grep -n -i -m {n} -e {pat} -- {rel_q}",
        ):
            try:
                result = sandbox.run(cmd)
            except Exception as exc:
                return f"error: {exc!r}"
            out = (result.stdout or "").strip()
            if out:
                if len(out) > GREP_FILE_OUTPUT_MAX:
                    out = out[:GREP_FILE_OUTPUT_MAX] + "\n...(truncated)"
                return out
            if result.returncode == 0:
                return "no matches"
        return "no matches"

    def search_hybrid(query: str, k: int = 5) -> str:
        """Search the corpus with BM25 + dense vectors (RRF fusion).

        Args:
            query: Natural-language question or keywords.
            k: Number of article hits to return (default 5).

        Returns:
            Ranked subjects with scores and relative paths under articles/. Then read_file
            or grep_file on a hit path; paginate read_file if the fact is not visible.
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
    if toolset in ("hybrid", "hybrid_deep", "full"):
        tools.append(search_hybrid)
    if toolset in ("shell", "full"):
        tools.append(run_shell)
    if toolset == "hybrid_deep":
        tools.append(grep_file)
    tools.extend([read_file, submit_answer])
    by_name = {t.__name__: t for t in tools}
    return tools, by_name
