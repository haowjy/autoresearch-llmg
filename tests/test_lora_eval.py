"""Tests for P1-02 LoRA eval row selection."""

from __future__ import annotations

import unittest
from pathlib import Path

from datasets import Dataset, DatasetDict

from llmg.train.run_lora_eval import adapter_dir_from_params, eval_rows_p1


class TestEvalRowsP1(unittest.TestCase):
    def test_start_row_offsets_eval_slice(self) -> None:
        ds = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": ["q0", "q1", "q2"],
                        "subject_sitelink": ["s0", "s1", "s2"],
                        "relation": ["r0", "r1", "r2"],
                        "object": ["a0", "a1", "a2"],
                        "snapshot_new": ["t0", "t1", "t2"],
                    }
                )
            }
        )

        questions, subjects, answers, as_of, row_indices = eval_rows_p1(
            ds,
            "test",
            max_rows=2,
            start_row=1,
        )

        self.assertEqual(questions, ["q1", "q2"])
        self.assertEqual(subjects, ["s1", "s2"])
        self.assertEqual(answers, ["a1", "a2"])
        self.assertEqual(as_of, ["t1", "t2"])
        self.assertEqual(row_indices, [1, 2])

    def test_start_row_past_end_returns_empty(self) -> None:
        ds = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": ["q0"],
                        "subject_sitelink": ["s0"],
                        "relation": ["r0"],
                        "object": ["a0"],
                        "snapshot_new": ["t0"],
                    }
                )
            }
        )

        rows = eval_rows_p1(ds, "test", max_rows=5, start_row=10)

        self.assertEqual(rows, ([], [], [], [], []))

    def test_relation_hint_appends_target_without_changing_gold_fields(self) -> None:
        ds = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": ["When did Subject join the group?"],
                        "subject_sitelink": ["Subject"],
                        "relation": ["member of"],
                        "object": ["The Group"],
                        "snapshot_new": ["2025-12-01"],
                    }
                )
            }
        )

        questions, subjects, answers, as_of, row_indices = eval_rows_p1(
            ds,
            "test",
            max_rows=1,
            relation_hint=True,
        )

        self.assertIn("When did Subject join the group?", questions[0])
        self.assertIn("Target relation: member of.", questions[0])
        self.assertIn("Return the full member of name/value for Subject.", questions[0])
        self.assertIn("submit the complete phrase from the question", questions[0])
        self.assertIn("not a supporting date/year", questions[0])
        self.assertEqual(subjects, ["Subject"])
        self.assertEqual(answers, ["The Group"])
        self.assertEqual(as_of, ["2025-12-01"])
        self.assertEqual(row_indices, [0])

    def test_when_nondate_hint_only_marks_when_questions_for_nondate_relations(self) -> None:
        ds = DatasetDict(
            {
                "test": Dataset.from_dict(
                    {
                        "question": [
                            "When did Subject join the group?",
                            "Which group did Subject join?",
                            "When did Subject start?",
                        ],
                        "subject_sitelink": ["Subject", "Subject", "Subject"],
                        "relation": ["member of", "member of", "start time"],
                        "object": ["The Group", "The Group", "2025"],
                        "snapshot_new": ["t0", "t1", "t2"],
                    }
                )
            }
        )

        questions, *_ = eval_rows_p1(
            ds,
            "test",
            max_rows=3,
            relation_hint="when_nondate",
        )

        self.assertIn("Target relation: member of.", questions[0])
        self.assertNotIn("Target relation:", questions[1])
        self.assertNotIn("Target relation:", questions[2])

    def test_adapter_dir_defaults_to_current_run(self) -> None:
        self.assertEqual(
            adapter_dir_from_params(Path("llmg/runs/new"), {}),
            Path("llmg/runs/new/lora_adapter"),
        )

    def test_adapter_dir_uses_relative_source_run(self) -> None:
        self.assertEqual(
            adapter_dir_from_params(
                Path("llmg/runs/new"),
                {"adapter_source_run": "20260529-165820_P1-02"},
            ),
            Path("llmg/runs/20260529-165820_P1-02/lora_adapter"),
        )

    def test_adapter_dir_accepts_explicit_adapter_path(self) -> None:
        self.assertEqual(
            adapter_dir_from_params(
                Path("llmg/runs/new"),
                {"adapter_source_run": "/tmp/source/lora_adapter"},
            ),
            Path("/tmp/source/lora_adapter"),
        )


if __name__ == "__main__":
    unittest.main()
