"""Unit tests for temporal retrieval and answer metrics (no GPU)."""

from __future__ import annotations

import unittest

from llmg.eval.temporal_metrics import (
    ANSWER_COSINE_HIT_THRESHOLD,
    answer_cosine_hit_rate,
    answer_exact_match,
    as_of_matches_slice_end,
    subject_recall_at_k,
    temporal_recall_at_k,
)
from llmg.agent.tools import normalize_submitted_answer, validate_submitted_answer
from llmg.data.corpus_export import dedupe_records, doc_id_for_row, DedupeStats, ArticleRecord


class TestTemporalRecall(unittest.TestCase):
    DOC_META = {
        "Conner_Kent__Nov_Dec": ("Conner Kent", "2020-11-01", "2020-12-01"),
        "Conner_Kent__Dec_Jan": ("Conner Kent", "2020-12-01", "2021-01-01"),
    }

    def test_wrong_slice_not_temporal_hit(self) -> None:
        self.assertFalse(
            temporal_recall_at_k(
                ["Conner_Kent__Nov_Dec"],
                gold_subject="Conner Kent",
                as_of="2021-01-01",
                doc_meta=self.DOC_META,
                k=5,
            )
        )

    def test_correct_slice_is_temporal_hit(self) -> None:
        self.assertTrue(
            temporal_recall_at_k(
                ["Conner_Kent__Dec_Jan"],
                gold_subject="Conner Kent",
                as_of="2021-01-01",
                doc_meta=self.DOC_META,
                k=5,
            )
        )

    def test_subject_recall_any_slice(self) -> None:
        self.assertTrue(
            subject_recall_at_k(["Conner Kent"], "Conner Kent", 5)
        )

    def test_as_of_matches_slice_end(self) -> None:
        self.assertTrue(
            as_of_matches_slice_end("2021-01-01", "2020-12-01", "2021-01-01")
        )
        self.assertFalse(
            as_of_matches_slice_end("2021-01-01", "2020-11-01", "2020-12-01")
        )


class TestAnswerTools(unittest.TestCase):
    def test_validate_rejects_prose(self) -> None:
        err = validate_submitted_answer("The article says Project Cadmus")
        self.assertIsNotNone(err)

    def test_validate_accepts_short_fact(self) -> None:
        self.assertIsNone(validate_submitted_answer("Project Cadmus"))

    def test_normalize_first_line(self) -> None:
        self.assertEqual(normalize_submitted_answer(' "CNN"\nextra'), "CNN")


class TestCorpusDedupe(unittest.TestCase):
    def test_doc_id_stable(self) -> None:
        self.assertEqual(
            doc_id_for_row("Conner Kent", "Nov→Dec"),
            doc_id_for_row("Conner Kent", "Nov→Dec"),
        )

    def test_dedupe_drops_duplicate_doc_id(self) -> None:
        r1 = ArticleRecord(
            doc_id="Foo__Nov_Dec",
            subject_sitelink="Foo",
            article="same",
            first_edited="",
            last_edited="",
            slice="Nov→Dec",
            split="train",
        )
        r2 = ArticleRecord(
            doc_id="Foo__Nov_Dec",
            subject_sitelink="Foo",
            article="same",
            first_edited="",
            last_edited="",
            slice="Nov→Dec",
            split="train",
        )
        out, stats = dedupe_records([r1, r2])
        self.assertEqual(len(out), 1)
        self.assertEqual(stats.dropped_duplicate_rows, 1)


class TestHarnessRecall(unittest.TestCase):
    def test_subject_recall_via_doc_meta_mapping(self) -> None:
        doc_meta = {"Doc_A": ("Gold Subject", "", "")}
        retrieved_subjects = [doc_meta["Doc_A"][0]]
        self.assertTrue(subject_recall_at_k(retrieved_subjects, "Gold Subject", 5))

    def test_cosine_hit_rate(self) -> None:
        rate = answer_cosine_hit_rate([0.9, 0.5], threshold=ANSWER_COSINE_HIT_THRESHOLD)
        self.assertAlmostEqual(rate, 0.5)


class TestAnswerEM(unittest.TestCase):
    def test_exact_match_normalized(self) -> None:
        self.assertEqual(answer_exact_match("CNN.", "cnn"), 1.0)


if __name__ == "__main__":
    unittest.main()
