"""Corpus export versioning and validation (filesystem, no dataset download)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from llmg.data.corpus_export import (
    ArticleRecord,
    corpus_is_current,
    export_filesystem,
    require_versioned_corpus,
    write_corpus_manifest,
    DedupeStats,
)
from llmg.memory.doc_catalog import DocCatalog


def _sample_records() -> list[ArticleRecord]:
    return [
        ArticleRecord(
            doc_id="Conner_Kent__Nov_Dec",
            subject_sitelink="Conner Kent",
            article="Article text.",
            first_edited="2020-11-01",
            last_edited="2020-12-01",
            slice="Nov→Dec",
            split="train",
        ),
    ]


class TestCorpusManifest(unittest.TestCase):
    def test_stale_corpus_without_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_filesystem(_sample_records(), root)
            with self.assertRaises(ValueError):
                require_versioned_corpus(root)

    def test_versioned_export_passes_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = _sample_records()
            export_filesystem(records, root)
            write_corpus_manifest(
                root,
                index_splits=["train"],
                stats=DedupeStats(1, 1, 0, 0),
            )
            self.assertTrue(corpus_is_current(root))
            require_versioned_corpus(root)
            catalog = DocCatalog.from_corpus_root(root)
            self.assertEqual(len(catalog), 1)
            self.assertEqual(
                catalog.meta_by_id["Conner_Kent__Nov_Dec"].last_edited,
                "2020-12-01",
            )

    def test_wrong_manifest_version_not_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_filesystem(_sample_records(), root)
            (root / "corpus_manifest.yaml").write_text(
                yaml.safe_dump({"corpus_version": 1}),
                encoding="utf-8",
            )
            self.assertFalse(corpus_is_current(root))


if __name__ == "__main__":
    unittest.main()
