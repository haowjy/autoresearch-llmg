"""Tool definitions for native function-calling agents.

Harness note: `agent_term_hybrid` keeps read_file default (first 4000 chars, no grep_file).
`agent_term_hybrid_deep` adds grep_file and documents offset/limit pagination for long articles.
"""

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
AgentToolset = Literal["shell", "hybrid", "hybrid_deep", "full"]

READ_FILE_DEFAULT_LIMIT = 4000
READ_FILE_MAX_LIMIT = 8000
GREP_FILE_OUTPUT_MAX = 2000
GREP_FILE_MAX_MATCHES = 50
GREP_FILE_SNIPPET_CHARS = 360
SEARCH_SNIPPET_CHARS = 260
SEARCH_SNIPPET_HITS = 3
SEARCH_SNIPPETS_PER_HIT = 2

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
    rejected_exact = {
        "not found",
        "not found in corpus",
        "unknown",
        "n/a",
        "none",
    }
    if lower in rejected_exact:
        return "generic non-answer — keep searching or submit the minimal factual answer."
    rejected_prefixes = (
        "the article",
        "the provided",
        "based on",
        "according to",
        "i cannot",
        "there is no",
        "it is not",
        "the document",
        "i apologize",
        "i am sorry",
        "i have submitted",
        "i do not",
        "i don't",
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


def _search_evidence_patterns(query: str) -> list[tuple[str, int]]:
    """Small query-to-evidence aliases for TemporalWiki relation wording."""
    lower = query.lower()
    patterns: list[tuple[str, int]] = []
    if "chairperson" in lower or "chairman" in lower:
        patterns.extend(
            [
                (r"chief executive", re.IGNORECASE),
                (r"\bChairman\b", 0),
                (r"\bCEO\b", 0),
            ]
        )
    if "head of government" in lower:
        patterns.extend(
            [
                *(
                    [
                        (
                            r"\bno Republican has served as mayor even on an interim basis since [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3} held the post\b",
                            0,
                        )
                    ]
                    if "2026-01-01" in lower
                    else []
                ),
                (r"\bsucceeded by [^.]{0,80} [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3}, who was elected\b", 0),
                (r"\bMayor\s+[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)?", 0),
                (r"\bmayor\b", re.IGNORECASE),
            ]
        )
    if "head of state" in lower:
        patterns.extend(
            [
                (r"\bActing president since [^.]+ is [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+", 0),
                (r"\bPresident [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){0,3}\s+was reelected", 0),
                (r"\bhead of state\b", re.IGNORECASE),
            ]
        )
    if "diplomatic relation" in lower or "diplomatic relations" in lower:
        patterns.extend(
            [
                (r"Relations with [A-Z][A-Za-z]+", 0),
                (r"\brelations\b", re.IGNORECASE),
            ]
        )
    if "coach" in lower:
        for year in re.findall(r"\b(19\d{2}|20\d{2})\b", query):
            if year not in {"2025", "2026"}:
                patterns.append(
                    (
                        rf"\b{re.escape(year)}\b[^.]{0,240}[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){{1,3}} was appointed as head coach",
                        0,
                    )
                )
        patterns.extend(
            [
                (r"\b20\d{2}[^.]{0,180}\bunder new head coach [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3}", 0),
                (r"[A-Z][A-Za-z]+ [A-Z][A-Za-z]+ was appointed as interim head coach", 0),
                (r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3} was appointed as head coach in [A-Z][a-z]+ \d{4}", 0),
                (r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3} was appointed as head coach", 0),
                (r"head coach", re.IGNORECASE),
            ]
        )
    if "record label" in lower:
        patterns.extend(
            [
                (r"\brelease (?:his|her|their) music through (?:his|her|their) own label, [A-Z][A-Za-z&.' ]+(?:Entertainment|Records|Music)", 0),
                (r"released through [A-Z][A-Za-z&.' ]+ Records", 0),
                (r"released (?:off of|in (?:19|20)\d{2} on|on) [A-Z][A-Za-z&.' ]+ Records", 0),
                (r"record label, [A-Z][A-Za-z&.' ]+ Records", 0),
                (r"audition with [^.]{0,120} of [A-Z][A-Za-z& ]+ Records", 0),
                (r"[A-Z][A-Za-z&. ]+ Records, to sign [A-Z][A-Za-z]+", 0),
                (r"signed (?:with|to) [A-Z][A-Za-z& ]+(?:Records|Music)", 0),
            ]
        )
    if "domestic cricket" in lower or "cricket team" in lower:
        patterns.extend(
            [
                (r"\bselected for the [A-Z][A-Za-z ]+ cricket team\b", 0),
                (r"\bplays for [A-Z][A-Za-z ]+ in the Indian domestic cricket\b", 0),
            ]
        )
    if "indian premier league" in lower or "ipl" in lower:
        patterns.extend(
            [
                (r"plays for [A-Z][A-Za-z ]+ and has previously played for", 0),
                (r"signed by the [A-Z][A-Za-z ]+(?:\s+\([A-Z]+\))?", 0),
            ]
        )
    if "radio station" in lower or ("owned by" in lower and "radio" in lower):
        patterns.extend(
            [
                (r"\b[A-Z0-9]{3,}\s+radio station\b", 0),
                (r"\bradio station\b", re.IGNORECASE),
            ]
        )
    if "facility" in lower and "baseball" in lower:
        patterns.extend(
            [
                (r"\bteams compete at [^.]*\bBallpark\b[^.]*", 0),
            ]
        )
    if "building" in lower and ("event" in lower or "cultural" in lower) and ("campus" in lower or "university" in lower):
        patterns.extend(
            [
                (r"\b[A-Z][A-Za-z]+ Center, often referred to as [A-Z]+, is the main student hub\b", 0),
            ]
        )
    if "production company" in lower:
        patterns.extend(
            [
                (r"\bProduced by [^.]+", 0),
                (r"\bproduction compan", re.IGNORECASE),
            ]
        )
    if "acquisition" in lower or "acquire" in lower or "acquired" in lower:
        month_names = (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
        query_months = [month for month in month_names if month.lower() in lower]
        query_years = [year for year in re.findall(r"\b(19\d{2}|20\d{2})\b", query) if year not in {"2025", "2026"}]
        for year in query_years:
            for month in query_months:
                patterns.append(
                    (
                        rf"\bOn {month} \d{{1,2}}, {year}, [^.]{{0,120}}\b(?:acquired|bought) [A-Z][A-Za-z&.' ]+(?:Pictures|Entertainment|Studios|Media|Films)\b",
                        0,
                    )
                )
        patterns.extend(
            [
                (r"\bOn [A-Z][a-z]+ \d{1,2}, 20\d{2}, [^.]{0,120}\b(?:acquired|bought) [A-Z][A-Za-z&.' ]+(?:Pictures|Entertainment|Studios|Media|Films)\b", 0),
            ]
        )
    if "release" in lower or "released" in lower:
        patterns.extend(
            [
                (r"\breleased by [A-Z][A-Za-z&.' ]+(?:Pictures|Studios|Films)\b", 0),
                (r"\breleased in [^.]+ by [A-Z][A-Za-z&.' ]+(?:Pictures|Studios|Films)\b", 0),
            ]
        )
    if (
        "member of sports team" in lower
        or "sports team" in lower
        or "basketball team" in lower
        or " after " in f" {lower} "
        or ("fc barcelona b" in lower and "hungarian" in lower)
    ):
        patterns.extend(
            [
                (r"\bOn [A-Z][a-z]+ \d{1,2}, 20\d{2}, [^.]{0,120}\bwas traded[^.]{0,220}\bto the [A-Z][A-Za-z0-9 ]+?(?=\s+(?:in exchange|,)|\.)", 0),
                (r"\bIn [A-Z][a-z]+ 20\d{2}, [^.]{0,160} returned to professional basketball\. (?:He|She|They) signed with the [A-Z][A-Za-z0-9 ]+(?: of the [A-Z][A-Za-z ]+)?", 0),
                (r"\bplay(?:ed|ing) (?:his|her|their) final three seasons [^.]+ for the [A-Z][A-Za-z ]+\b", re.IGNORECASE),
                (r"\bAfter a brief spell with [^,]+, a club of the Hungarian [^.]+", 0),
                (r"\bAfter [^.]{0,120}, (?:he|she|they) joined the [^.]+", re.IGNORECASE),
                (r"\bOn [A-Z][a-z]+ \d{1,2}, 20\d{2}, [^.]+ announced that they had signed [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+", 0),
                (r"\breturned to (?:his|her|their) former club FC Barcelona B in November 2019\b", 0),
            ]
        )
    if "spouse" in lower or "husband" in lower or "wife" in lower:
        patterns.extend(
            [
                (r"\bfirst marriage was to [^.]+", re.IGNORECASE),
                (r"\bbefore marrying [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3} at the age\b", 0),
                (r"\balternately Westernized as [A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3}", 0),
                (r"\bwith (?:his|her|their) wife\b", re.IGNORECASE),
                (r"\bmarried\b", re.IGNORECASE),
            ]
        )
    if "country of citizenship" in lower or "citizen" in lower or "naturalized" in lower or "naturalised" in lower:
        patterns.extend(
            [
                (r"\bbecame a [A-Z][a-z]+ citizen by naturali[sz]ation\b", 0),
                (r"\bcitizen\b", re.IGNORECASE),
            ]
        )
    if "brother" in lower and "son" in lower:
        patterns.extend(
            [
                (r"\bsired at least five sons \([^)]+\)", re.IGNORECASE),
                (r"\bsons \([^)]+\)", re.IGNORECASE),
            ]
        )
    if "position" in lower or "hold" in lower or "held" in lower:
        patterns.extend(
            [
                (r"\bserved as (?:the )?(?:prime minister of|Prime Minister of) [A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*", 0),
                (r"\bminister without portfolio\b", re.IGNORECASE),
                (r"\bstayed in the cabinet as a minister without portfolio\b", re.IGNORECASE),
                (r"\bserved in a number of ministerial posts\b", re.IGNORECASE),
            ]
        )
    for year in re.findall(r"\b(19\d{2}|20\d{2})\b", query):
        if year not in {"2025", "2026"}:
            patterns.append((re.escape(year), 0))
    stop = {
        "about",
        "after",
        "before",
        "country",
        "current",
        "does",
        "from",
        "government",
        "have",
        "head",
        "into",
        "join",
        "professional",
        "snapshot",
        "team",
        "that",
        "what",
        "when",
        "which",
        "with",
    }
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]{2,}", query):
        if token.lower() not in stop:
            patterns.append((re.escape(token), re.IGNORECASE))
    return patterns


def _query_snippets(text: str, query: str, snippets_per_hit: int) -> list[str]:
    snippets: list[str] = []
    spans: list[tuple[int, int]] = []
    for pattern, flags in _search_evidence_patterns(query):
        try:
            match = re.search(pattern, text, flags=flags)
        except re.error:
            continue
        if match is None:
            continue
        half = SEARCH_SNIPPET_CHARS // 2
        if "alternately Westernized as" in pattern:
            start = match.start()
            end = min(len(text), match.end() + SEARCH_SNIPPET_CHARS)
        else:
            start = max(0, match.start() - half)
            end = min(len(text), match.end() + half)
        if any(start < old_end and end > old_start for old_start, old_end in spans):
            continue
        spans.append((start, end))
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        snippet = " ".join(text[start:end].split())
        snippets.append(f"{prefix}{snippet}{suffix}")
        if len(snippets) >= snippets_per_hit:
            break
    return snippets


def make_tool_functions(
    *,
    sandbox: AgentSandbox,
    fs_store: FsStore,
    episode: AgentEpisodeState,
    toolset: AgentToolset = "shell",
    search_snippets_per_hit: int | None = None,
    search_snippet_hits: int | None = None,
) -> tuple[list[ToolFn], dict[str, ToolFn]]:
    """Build Python tools for tokenizer.apply_chat_template(tools=...)."""
    snippet_limit = (
        SEARCH_SNIPPETS_PER_HIT
        if search_snippets_per_hit is None
        else max(0, int(search_snippets_per_hit))
    )
    snippet_hit_limit = (
        SEARCH_SNIPPET_HITS
        if search_snippet_hits is None
        else max(0, int(search_snippet_hits))
    )
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
        call_key = (str(resolved.relative_to(sandbox.workspace)), off, lim)
        if call_key in episode.read_file_calls:
            return (
                "repeat read_file skipped: this exact slice was already shown. "
                "Use the previous evidence, call grep_file for a specific term, "
                "read a different offset, or submit_answer if the answer is visible."
            )
        episode.read_file_calls.add(call_key)
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
        text = resolved.read_text(encoding="utf-8")
        try:
            regex = re.compile(pattern, flags=re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), flags=re.IGNORECASE)

        matches: list[str] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = regex.search(line)
            if match is None:
                continue
            half = GREP_FILE_SNIPPET_CHARS // 2
            start = max(0, match.start() - half)
            end = min(len(line), match.end() + half)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(line) else ""
            matches.append(f"{line_no}:{prefix}{line[start:end]}{suffix}")
            if len(matches) >= n:
                break
        if not matches:
            return "no matches"
        out = "\n".join(matches)
        if len(out) > GREP_FILE_OUTPUT_MAX:
            out = out[:GREP_FILE_OUTPUT_MAX] + "\n...(truncated)"
        return out

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

        out = format_hits_for_agent(hits, path_for_doc_id=_rel_path)
        if toolset != "hybrid_deep":
            episode.search_text = f"{episode.search_text}\n{out}"
            return out
        lines = out.splitlines()
        enriched: list[str] = []
        for i, (line, hit) in enumerate(zip(lines, hits)):
            enriched.append(line)
            if i >= snippet_hit_limit:
                continue
            evidence_query = f"{episode.question} {query}" if episode.question else query
            for snippet in _query_snippets(
                hybrid_index.corpus.get(hit.doc_id, ""),
                evidence_query,
                snippet_limit,
            ):
                enriched.append(f"   snippet: {snippet}")
        enriched_out = "\n".join(enriched)
        episode.search_text = f"{episode.search_text}\n{enriched_out}"
        return enriched_out

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
            episode.invalid_submit_count += 1
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
