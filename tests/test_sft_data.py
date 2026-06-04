"""Tests for TemporalWiki SFT message formatting."""

from __future__ import annotations

import unittest

from datasets import Dataset

from llmg.train.sft_data import build_sft_dataset, row_to_messages


class TestSftData(unittest.TestCase):
    def test_row_to_messages_includes_article(self) -> None:
        row = {
            "question": "Who is X as of 2025-12-01?",
            "article": "X is a member of Org Y.",
            "object": "Org Y",
        }
        msgs = row_to_messages(row, max_article_chars=100)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertIn("Who is X", msgs[0]["content"])
        self.assertIn("Article:", msgs[0]["content"])
        self.assertIn("Org Y", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "Org Y")

    def test_truncates_article(self) -> None:
        row = {"question": "Q?", "article": "a" * 500, "object": "ans"}
        msgs = row_to_messages(row, max_article_chars=10)
        self.assertLessEqual(len(msgs[0]["content"]), 10 + len("Q?\n\nArticle:\n"))

    def test_concise_format_adds_answer_instruction(self) -> None:
        row = {"question": "Q?", "article": "A.", "object": "ans"}
        msgs = row_to_messages(row, max_article_chars=100, sft_format="concise")
        self.assertIn("Answer with only the short factual answer.", msgs[0]["content"])
        self.assertEqual(msgs[1]["content"], "ans")

    def test_tool_trace_format_emits_native_tool_trajectory(self) -> None:
        row = {
            "subject": "Conner Kent",
            "relation": "member of",
            "object": "Project Cadmus",
            "subject_sitelink": "Conner Kent",
            "question": "Who is Conner Kent's member of as of 2025-12-01?",
            "slice": "Nov->Dec",
            "snapshot_new": "2025-12-01",
            "article": "Conner Kent was created by Project Cadmus in this version.",
        }
        msgs = row_to_messages(row, max_article_chars=100, sft_format="tool_trace")

        self.assertEqual([m["role"] for m in msgs], [
            "system",
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant",
        ])
        self.assertEqual(
            msgs[2]["tool_calls"][0]["function"],
            {
                "name": "search_hybrid",
                "arguments": {"query": "Conner Kent member of"},
            },
        )
        self.assertIn("path=articles/Conner_Kent__Nov_Dec.md", msgs[3]["content"])
        self.assertEqual(
            msgs[4]["tool_calls"][0]["function"],
            {
                "name": "read_file",
                "arguments": {"path": "articles/Conner_Kent__Nov_Dec.md"},
            },
        )
        self.assertIn("Project Cadmus", msgs[5]["content"])
        self.assertEqual(
            msgs[6]["tool_calls"][0]["function"],
            {
                "name": "submit_answer",
                "arguments": {"answer": "Project Cadmus"},
            },
        )

    def test_unknown_format_raises(self) -> None:
        row = {"question": "Q?", "article": "A.", "object": "ans"}
        with self.assertRaises(ValueError):
            row_to_messages(row, sft_format="unknown")  # type: ignore[arg-type]

    def test_mixed_tool_trace_dataset_doubles_rows(self) -> None:
        split = Dataset.from_dict(
            {
                "subject": ["Conner Kent"],
                "relation": ["member of"],
                "object": ["Project Cadmus"],
                "subject_sitelink": ["Conner Kent"],
                "question": ["Who is Conner Kent's member of as of 2025-12-01?"],
                "slice": ["Nov->Dec"],
                "snapshot_new": ["2025-12-01"],
                "article": ["Conner Kent was created by Project Cadmus."],
            }
        )

        ds = build_sft_dataset(split, sft_format="mixed_tool_trace")

        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]["messages"][-1]["content"], "Project Cadmus")
        self.assertEqual(
            ds[1]["messages"][-1]["tool_calls"][0]["function"]["name"],
            "submit_answer",
        )


if __name__ == "__main__":
    unittest.main()
