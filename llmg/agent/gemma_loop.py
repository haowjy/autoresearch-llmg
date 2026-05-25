"""Multi-turn agentic search loop — Gemma 4 native chat template + tools."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llmg.agent.sandbox import AgentSandbox
from llmg.agent.state import AgentEpisodeState
from llmg.agent.trace import (
    TRACE_RAW_MAX,
    TRACE_TOOL_MAX,
    end_episode_trace,
    start_episode_trace,
    truncate_text,
    write_trace,
)
from llmg.agent.tools import (
    AgentToolset,
    execute_parsed_tool_calls,
    make_tool_functions,
    supports_native_tools,
)
from llmg.data.corpus_export import require_versioned_corpus
from llmg.eval.temporal_metrics import (
    answer_cosine_hit_rate,
    answer_cosine_similarity_batch,
    answer_exact_match,
    subject_recall_at_k,
    temporal_recall_at_k,
)
from llmg.memory.doc_catalog import DocCatalog
from llmg.memory.fs_store import FsStore

log = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-4-E4B-it"
_DEFAULT_TOOL_MSG_MAX_CHARS = 2000  # probe: 6000 OOM on 24GB @ 16 steps (rows 5–8)
_DEFAULT_MAX_NEW_TOKENS = 384


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _tool_msg_max_chars() -> int:
    return _env_int("LLMG_AGENT_TOOL_MSG_MAX", _DEFAULT_TOOL_MSG_MAX_CHARS)


def _max_new_tokens() -> int:
    return _env_int("LLMG_AGENT_MAX_NEW_TOKENS", _DEFAULT_MAX_NEW_TOKENS)


def _empty_cache_after_episode() -> bool:
    return os.environ.get("LLMG_AGENT_EMPTY_CACHE_EPISODE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )

# Minimal system text — tool schemas carry mechanics; one optional final-step nudge only.
AGENTIC_SYSTEM_SHELL = (
    "You are a research assistant with tools. The corpus is under articles/ (Markdown + YAML dates)."
)
AGENTIC_SYSTEM_HYBRID = (
    "You are a research assistant with tools. The corpus is under articles/ (Markdown + YAML dates)."
)

# One synthetic user turn when the episode is almost out of steps (not mid-loop coaching).
FINAL_STEP_NUDGE = "Answer with submit_answer. You are taking too long."


@dataclass
class AgentStepTrace:
    step: int
    response: str
    tool_call: dict[str, Any] | None
    observation: str


@dataclass
class AgentRunResult:
    answer: str
    retrieved_doc_ids: list[str]
    steps: int
    traces: list[AgentStepTrace] = field(default_factory=list)
    cmd_count: int = 0
    bytes_read: int = 0


def _clean_answer(content: str | None) -> str:
    if not content:
        return ""
    text = content.strip()
    if text in ("<eos>", "<|eos|>", "<|file_separator|>"):
        return ""
    if "<tool_call>" in text or "<|tool_call>" in text:
        return ""
    text = re.sub(r"<\|[^|]+\|>[^<]*", "", text)
    text = re.sub(r"<\|channel\|>\s*\w+\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


class GemmaAgentLoop:
    """Gemma 4 agent: conversation `messages` list grows each turn (chat history stack)."""

    def __init__(
        self,
        *,
        corpus_root: Path | None = None,
        model_name: str | None = None,
        max_steps: int = 8,
        agent_toolset: AgentToolset = "shell",
        tool_msg_max_chars: int | None = None,
        max_new_tokens: int | None = None,
        empty_cache_after_episode: bool | None = None,
    ) -> None:
        self.default_corpus_root = corpus_root
        self.model_name = model_name or os.environ.get("LLMG_AGENT_MODEL", DEFAULT_MODEL)
        self.max_steps = max_steps
        self.agent_toolset = agent_toolset
        self.tool_msg_max_chars = (
            tool_msg_max_chars if tool_msg_max_chars is not None else _tool_msg_max_chars()
        )
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else _max_new_tokens()
        )
        self.empty_cache_after_episode = (
            empty_cache_after_episode
            if empty_cache_after_episode is not None
            else _empty_cache_after_episode()
        )
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        log.info("loading agent model %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if not supports_native_tools(self._tokenizer):
            raise RuntimeError(
                f"Model {self.model_name!r} has no native tool chat template. "
                "Use a Gemma 4 instruction-tuned checkpoint (e.g. google/gemma-4-E4B-it)."
            )

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def _generate(self, messages: list[dict], tools: list) -> str:
        """One ``model.generate`` per turn (full chat template each step)."""
        import torch

        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        device = self._model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_token_ids = out[0, prompt_len:].detach().cpu().tolist()
        return self._tokenizer.decode(new_token_ids, skip_special_tokens=False)

    def _parse_response(self, text: str) -> dict[str, Any]:
        assert self._tokenizer is not None
        try:
            return self._tokenizer.parse_response(text)
        except Exception as exc:
            log.debug("parse_response failed: %s raw=%r", exc, text[:200])
            return {"role": "assistant", "content": _clean_answer(text)}

    def _system_prompt(self) -> str:
        if self.agent_toolset in ("hybrid", "full"):
            return AGENTIC_SYSTEM_HYBRID
        return AGENTIC_SYSTEM_SHELL

    def _build_messages(self, question: str, as_of: str) -> list[dict[str, Any]]:
        user = question if not as_of else f"{question}\n\n(as-of: {as_of})"
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user},
        ]

    def _run_agentic_loop(
        self,
        messages: list[dict[str, Any]],
        *,
        sandbox: AgentSandbox,
        fs_store: FsStore,
        retrieved_doc_ids: list[str],
        traces: list[AgentStepTrace],
        trace_path: Path | None,
        episode: AgentEpisodeState,
    ) -> str:
        """Each turn: chat-template prompt → model.generate → append to messages."""
        tool_list, tools_by_name = make_tool_functions(
            sandbox=sandbox,
            fs_store=fs_store,
            episode=episode,
            toolset=self.agent_toolset,
        )
        answer = ""
        final_nudge_sent = False

        for step in range(self.max_steps):
            if (
                not episode.done
                and not final_nudge_sent
                and step >= self.max_steps - 2
            ):
                messages.append({"role": "user", "content": FINAL_STEP_NUDGE})
                write_trace(
                    trace_path,
                    "user_nudge",
                    step=step,
                    content=FINAL_STEP_NUDGE,
                    kind="final_step",
                )
                final_nudge_sent = True

            raw = self._generate(messages, tool_list)
            parsed = self._parse_response(raw)
            tool_calls = parsed.get("tool_calls") or []
            content = _clean_answer(parsed.get("content"))

            write_trace(
                trace_path,
                "assistant_turn",
                step=step,
                messages_len=len(messages),
                raw=truncate_text(raw, TRACE_RAW_MAX),
                tool_calls=tool_calls,
                content=content,
            )

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": tool_calls,
                    }
                )
                tool_msgs = execute_parsed_tool_calls(
                    tool_calls,
                    sandbox=sandbox,
                    fs_store=fs_store,
                    retrieved_doc_ids=retrieved_doc_ids,
                    tools_by_name=tools_by_name,
                    episode=episode,
                )
                for m in tool_msgs:
                    body = m.get("content", "")
                    if len(body) > self.tool_msg_max_chars:
                        m = {**m, "content": truncate_text(body, self.tool_msg_max_chars)}
                    messages.append(m)
                    write_trace(
                        trace_path,
                        "tool_result",
                        step=step,
                        name=m.get("name", ""),
                        content=truncate_text(str(m.get("content", "")), TRACE_TOOL_MAX),
                    )
                obs = "\n---\n".join(m["content"] for m in tool_msgs)
                traces.append(
                    AgentStepTrace(
                        step=step,
                        response=raw[:2000],
                        tool_call={"tool_calls": tool_calls},
                        observation=obs[:4000],
                    )
                )
                if episode.done and episode.final_answer:
                    answer = episode.final_answer
                    break
                continue

            messages.append({"role": "assistant", "content": content or None})
            traces.append(
                AgentStepTrace(
                    step=step,
                    response=raw[:2000],
                    tool_call=None,
                    observation="(no tool calls)",
                )
            )

        if not answer and episode.final_answer:
            answer = episode.final_answer
        return answer

    def run_query(
        self,
        question: str,
        *,
        as_of: str = "",
        corpus_root: Path | None = None,
        trace_path: Path | None = None,
        gold_subject: str = "",
        gold_answer: str = "",
        row_index: int | None = None,
    ) -> AgentRunResult:
        root = corpus_root or self.default_corpus_root
        if root is None:
            raise ValueError("corpus_root required")

        start_episode_trace(
            trace_path,
            question=question,
            as_of=as_of,
            model=self.model_name,
            toolset=self.agent_toolset,
            gold_subject=gold_subject,
            gold_answer=gold_answer,
            row_index=row_index,
        )
        self.fs = FsStore(root)
        sandbox = AgentSandbox(root, trace_path=trace_path)
        retrieved_doc_ids: list[str] = []
        traces: list[AgentStepTrace] = []
        episode = AgentEpisodeState()

        self._load_model()
        messages = self._build_messages(question, as_of)
        answer = self._run_agentic_loop(
            messages,
            sandbox=sandbox,
            fs_store=self.fs,
            retrieved_doc_ids=retrieved_doc_ids,
            traces=traces,
            trace_path=trace_path,
            episode=episode,
        )

        sandbox.close()
        import torch

        if torch.cuda.is_available() and self.empty_cache_after_episode:
            torch.cuda.empty_cache()

        end_episode_trace(
            trace_path,
            answer=answer,
            retrieved_doc_ids=retrieved_doc_ids,
            steps=len(traces),
            cmd_count=sandbox.stats.cmd_count,
            bytes_read=sandbox.stats.bytes_read,
        )
        return AgentRunResult(
            answer=answer,
            retrieved_doc_ids=retrieved_doc_ids,
            steps=len(traces),
            traces=traces,
            cmd_count=sandbox.stats.cmd_count,
            bytes_read=sandbox.stats.bytes_read,
        )


def run_agent_eval(
    *,
    corpus_root: Path,
    questions: list[str],
    gold_subjects: list[str],
    gold_answers: list[str],
    as_of_dates: list[str],
    k: int = 5,
    max_steps: int = 8,
    max_rows: int | None = None,
    model_name: str | None = None,
    trace_dir: Path | None = None,
    agent_toolset: AgentToolset = "shell",
) -> dict[str, float]:
    n = len(questions)
    if max_rows is not None:
        n = min(n, max_rows)
    if n == 0:
        return {
            f"retrieval_recall@{k}": 0.0,
            f"temporal_recall@{k}": 0.0,
            "answer_em": 0.0,
            "answer_cosine": 0.0,
            "answer_cosine_hit_rate": 0.0,
        }

    require_versioned_corpus(corpus_root)
    catalog = DocCatalog.from_corpus_root(corpus_root)
    doc_meta = catalog.doc_meta_tuple()

    agent = GemmaAgentLoop(
        corpus_root=corpus_root,
        model_name=model_name,
        max_steps=max_steps,
        agent_toolset=agent_toolset,
    )

    recall_hits = 0
    temporal_hits = 0
    em_hits = 0.0
    total_steps = 0
    total_cmds = 0
    total_bytes = 0
    results: list[AgentRunResult] = []

    import torch

    for i in range(n):
        trace_path = trace_dir / f"row_{i}.jsonl" if trace_dir else None
        result = agent.run_query(
            questions[i],
            as_of=as_of_dates[i] if i < len(as_of_dates) else "",
            trace_path=trace_path,
            gold_subject=gold_subjects[i],
            gold_answer=gold_answers[i],
            row_index=i,
        )
        results.append(result)
        gold = gold_subjects[i]
        as_of = as_of_dates[i] if i < len(as_of_dates) else ""
        subjects = [
            doc_meta[d][0] for d in result.retrieved_doc_ids[:k] if d in doc_meta
        ]
        if subject_recall_at_k(subjects, gold, k):
            recall_hits += 1
        if temporal_recall_at_k(
            result.retrieved_doc_ids,
            gold_subject=gold,
            as_of=as_of,
            doc_meta=doc_meta,
            k=k,
        ):
            temporal_hits += 1
        em_hits += answer_exact_match(result.answer, gold_answers[i])
        total_steps += result.steps
        total_cmds += result.cmd_count
        total_bytes += result.bytes_read
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    cos_scores = answer_cosine_similarity_batch(
        [(r.answer, gold_answers[i]) for i, r in enumerate(results)]
    )
    cos_mean = sum(cos_scores) / n if cos_scores else 0.0

    return {
        f"retrieval_recall@{k}": recall_hits / n,
        f"temporal_recall@{k}": temporal_hits / n,
        "answer_em": em_hits / n,
        "answer_cosine": cos_mean,
        "answer_cosine_hit_rate": answer_cosine_hit_rate(cos_scores),
        "agent_steps_mean": total_steps / n,
        "cmd_count_mean": total_cmds / n,
        "bytes_read_mean": total_bytes / n,
        "eval_rows": float(n),
    }
