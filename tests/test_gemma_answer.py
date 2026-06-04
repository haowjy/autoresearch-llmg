"""Tests for agent answer extraction and fallback."""

from __future__ import annotations

import os
import unittest

from llmg.agent.gemma_loop import (
    AgentRunResult,
    _answer_subset_metrics,
    _answer_named_in_question,
    _clean_answer,
    _embedded_submit_answer_tool_calls,
    _resolve_episode_answer,
)
from llmg.agent.state import AgentEpisodeState


class TestCleanAnswer(unittest.TestCase):
    def test_strips_turn_token(self) -> None:
        self.assertEqual(_clean_answer("The New York Times<turn|>"), "The New York Times")

    def test_empty_tool_call_fragment(self) -> None:
        self.assertEqual(_clean_answer("<|tool_call|>foo"), "")

    def test_empty_malformed_tool_call_text(self) -> None:
        self.assertEqual(_clean_answer("submit_answer{answer:Stellantis}<tool_call|><eos>"), "")


class TestEmbeddedSubmitAnswer(unittest.TestCase):
    def test_extracts_canonical_embedded_submit(self) -> None:
        calls = _embedded_submit_answer_tool_calls(
            "Based on the snippet.\n"
            "<|tool_call>call:submit_answer{answer:<|\"|>Los Angeles Clippers<|\"|>}<tool_call|><eos>"
        )
        self.assertEqual(
            calls,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_answer",
                        "arguments": {"answer": "Los Angeles Clippers"},
                    },
                }
            ],
        )

    def test_ignores_unquoted_malformed_submit(self) -> None:
        self.assertEqual(
            _embedded_submit_answer_tool_calls("submit_answer{answer:Stellantis}<tool_call|><eos>"),
            [],
        )


class TestResolveEpisodeAnswer(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("LLMG_AGENT_FALLBACK_LAST_ANSWER", None)
        os.environ.pop("LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK", None)

    def test_submit_wins(self) -> None:
        ep = AgentEpisodeState(final_answer="Project Cadmus", done=True)
        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="The New York Times",
        )
        self.assertEqual(ans, "Project Cadmus")
        self.assertEqual(src, "submit_answer")

    def test_fallback_when_enabled(self) -> None:
        os.environ["LLMG_AGENT_FALLBACK_LAST_ANSWER"] = "1"
        ep = AgentEpisodeState()
        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="The New York Times",
        )
        self.assertEqual(ans, "The New York Times")
        self.assertEqual(src, "last_content_fallback")

    def test_no_fallback_when_disabled(self) -> None:
        os.environ["LLMG_AGENT_FALLBACK_LAST_ANSWER"] = "0"
        ep = AgentEpisodeState()
        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="The New York Times",
        )
        self.assertEqual(ans, "")
        self.assertEqual(src, "none")

    def test_no_fallback_for_invalid_non_answer_text(self) -> None:
        ep = AgentEpisodeState()
        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="I apologize, but I cannot find this in the corpus.",
        )
        self.assertEqual(ans, "")
        self.assertEqual(src, "none")

    def test_no_fallback_for_malformed_submit_text(self) -> None:
        ep = AgentEpisodeState()
        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content=_clean_answer("submit_answer{answer:Stellantis}<tool_call|><eos>"),
        )
        self.assertEqual(ans, "")
        self.assertEqual(src, "none")

    def test_search_candidate_fallback_overrides_unseen_wrong_submit(self) -> None:
        os.environ["LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK"] = "1"
        ep = AgentEpisodeState(
            final_answer="Sam Adams",
            done=True,
            as_of="2025-12-01",
            question="Who was the head of government in Portland?",
            search_text="snippet: Mayor Neil Goldschmidt took office in 1972.",
        )

        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="",
        )

        self.assertEqual(ans, "Neil Goldschmidt")
        self.assertEqual(src, "search_candidate_fallback")

    def test_search_candidate_fallback_does_not_override_future_snapshot_answer(self) -> None:
        os.environ["LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK"] = "1"
        ep = AgentEpisodeState(
            final_answer="Armenia",
            done=True,
            as_of="2026-01-01",
            question="What country is Pakistan in a diplomatic relation with as of 2026-01-01?",
            search_text="snippet: Relations with Russia have improved since the end of the Cold War.",
        )

        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="",
        )

        self.assertEqual(ans, "Armenia")
        self.assertEqual(src, "submit_answer")

    def test_search_candidate_fallback_allows_safe_future_snapshot_support(self) -> None:
        os.environ["LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK"] = "1"
        ep = AgentEpisodeState(
            final_answer="Charlie Hales",
            done=True,
            as_of="2026-01-01",
            question="Who was the head of government in Portland, Oregon as of 2026-01-01?",
            search_text=(
                "snippet: no Republican has served as mayor even on an interim basis since "
                "Connie McCready held the post from 1979 to 1980."
            ),
        )

        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="",
        )

        self.assertEqual(ans, "Connie McCready")
        self.assertEqual(src, "search_candidate_fallback")

    def test_search_candidate_fallback_allows_domestic_cricket_full_surface(self) -> None:
        os.environ["LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK"] = "1"
        ep = AgentEpisodeState(
            final_answer="Mumbai",
            done=True,
            as_of="2026-01-01",
            question="Which cricket team did Suryakumar Yadav represent in domestic cricket as of August 2024?",
            search_text=(
                "snippet: Yadav started playing club cricket in Mumbai before he was "
                "selected for the Mumbai cricket team. Yadav plays for Mumbai in "
                "the Indian domestic cricket."
            ),
        )

        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="",
        )

        self.assertEqual(ans, "Mumbai cricket team")
        self.assertEqual(src, "search_candidate_fallback")

    def test_search_candidate_fallback_allows_midseason_trade_support(self) -> None:
        os.environ["LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK"] = "1"
        ep = AgentEpisodeState(
            final_answer="New York Knicks",
            done=True,
            as_of="2026-01-01",
            question=(
                "Which professional basketball team did James Donaldson play for "
                "during the 1991–92 season?"
            ),
            search_text=(
                "snippet: After brief stints with the New York Knicks "
                "(traded midway through 1991–92 for Brian Quinnett) and Utah Jazz "
                "(49 games in two seasons combined) in the early 1990s."
            ),
        )

        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="",
        )

        self.assertEqual(ans, "Utah Jazz")
        self.assertEqual(src, "search_candidate_fallback")

    def test_search_candidate_fallback_does_not_override_answer_named_in_question(self) -> None:
        os.environ["LLMG_AGENT_SEARCH_CANDIDATE_FALLBACK"] = "1"
        ep = AgentEpisodeState(
            final_answer="Shadow Chancellor of the Exchequer",
            done=True,
            question="When did Harold Wilson hold the position of Shadow Chancellor of the Exchequer?",
            search_text="snippet: served as Prime Minister of the United Kingdom.",
        )

        ans, src = _resolve_episode_answer(
            loop_answer="",
            episode=ep,
            last_assistant_content="",
        )

        self.assertEqual(ans, "Shadow Chancellor of the Exchequer")
        self.assertEqual(src, "submit_answer")


class TestAnswerNamedInQuestion(unittest.TestCase):
    def test_exact_phrase_named_in_question(self) -> None:
        self.assertTrue(
            _answer_named_in_question(
                "When did Harold Wilson hold the position of Shadow Chancellor of the Exchequer?",
                "Shadow Chancellor of the Exchequer",
            )
        )

    def test_absent_phrase_not_named_in_question(self) -> None:
        self.assertFalse(_answer_named_in_question("Who was the mayor?", "Neil Goldschmidt"))


class TestAnswerSubsetMetrics(unittest.TestCase):
    def test_submitted_and_answered_only_metrics(self) -> None:
        results = [
            AgentRunResult(
                answer="Project Cadmus",
                retrieved_doc_ids=[],
                steps=1,
                answer_source="submit_answer",
            ),
            AgentRunResult(
                answer="The Longshot",
                retrieved_doc_ids=[],
                steps=1,
                answer_source="last_content_fallback",
            ),
            AgentRunResult(
                answer="",
                retrieved_doc_ids=[],
                steps=1,
                answer_source="none",
                invalid_submit_count=2,
            ),
        ]
        metrics = _answer_subset_metrics(
            results,
            cos_scores=[1.0, 0.2, 0.0],
            em_scores=[1.0, 0.0, 0.0],
        )

        self.assertEqual(metrics["answer_answered_only_count"], 2.0)
        self.assertEqual(metrics["answer_answered_only_em"], 0.5)
        self.assertEqual(metrics["answer_answered_only_cosine"], 0.6)
        self.assertEqual(metrics["answer_answered_only_cosine_hit_rate"], 0.5)
        self.assertEqual(metrics["answer_submitted_only_count"], 1.0)
        self.assertEqual(metrics["answer_submitted_only_em"], 1.0)
        self.assertEqual(metrics["answer_submitted_only_cosine"], 1.0)
        self.assertEqual(metrics["answer_submitted_only_cosine_hit_rate"], 1.0)


class TestAgentRunResult(unittest.TestCase):
    def test_invalid_submit_count_defaults_zero(self) -> None:
        result = AgentRunResult(answer="", retrieved_doc_ids=[], steps=0)
        self.assertEqual(result.invalid_submit_count, 0)


if __name__ == "__main__":
    unittest.main()
