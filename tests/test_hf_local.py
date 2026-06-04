"""Tests for HF local-only helpers."""

from __future__ import annotations

import os
import unittest

from llmg.search.embeddings import clear_sentence_embedder_cache, get_sentence_embedder
from llmg.util.hf_local import configure_hf_offline_if_requested, hf_local_files_only


class TestHfLocal(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("LLMG_HF_LOCAL_ONLY", None)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(key, None)
        clear_sentence_embedder_cache()

    def test_hf_local_files_only_false_by_default(self) -> None:
        os.environ.pop("LLMG_HF_LOCAL_ONLY", None)
        self.assertFalse(hf_local_files_only())

    def test_hf_local_files_only_true(self) -> None:
        os.environ["LLMG_HF_LOCAL_ONLY"] = "1"
        self.assertTrue(hf_local_files_only())

    def test_configure_sets_offline_env(self) -> None:
        os.environ["LLMG_HF_LOCAL_ONLY"] = "yes"
        configure_hf_offline_if_requested()
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
        self.assertEqual(os.environ.get("HF_DATASETS_OFFLINE"), "1")

    def test_shared_embedder_singleton_per_model(self) -> None:
        from unittest.mock import MagicMock, patch

        clear_sentence_embedder_cache()
        mock_cls = MagicMock(side_effect=lambda *a, **k: MagicMock(name="embedder"))
        with patch("sentence_transformers.SentenceTransformer", mock_cls):
            a = get_sentence_embedder("sentence-transformers/all-MiniLM-L6-v2")
            b = get_sentence_embedder("sentence-transformers/all-MiniLM-L6-v2")
        self.assertIs(a, b)
        mock_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
