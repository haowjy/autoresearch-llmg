"""Mutable state for one agentic episode (single question)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentEpisodeState:
    final_answer: str | None = None
    done: bool = False
