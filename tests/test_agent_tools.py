"""Unit tests for agent read_file pagination and grep_file path validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmg.agent.sandbox import AgentSandbox
from llmg.agent.state import AgentEpisodeState
from llmg.agent.tools import READ_FILE_DEFAULT_LIMIT, make_tool_functions
from llmg.memory.fs_store import FsStore


class TestReadFilePagination(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        articles = self.workspace / "articles"
        articles.mkdir(parents=True)
        self.body = "HEAD" + ("x" * 5000) + "TAIL_MARKER" + ("y" * 500)
        (articles / "sample.md").write_text(self.body, encoding="utf-8")
        self.sandbox = AgentSandbox(self.workspace)
        self.fs = FsStore(self.workspace)
        self.episode = AgentEpisodeState()
        _, self.tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_default_first_slice(self) -> None:
        out = self.tools["read_file"](path="articles/sample.md")
        self.assertEqual(out, self.body[:READ_FILE_DEFAULT_LIMIT])
        self.assertIn("HEAD", out)
        self.assertNotIn("TAIL_MARKER", out)

    def test_offset_limit_slice(self) -> None:
        off = self.body.index("TAIL_MARKER")
        out = self.tools["read_file"](path="articles/sample.md", offset=off, limit=50)
        self.assertTrue(out.startswith("TAIL_MARKER"))
        self.assertLessEqual(len(out), 50)

    def test_offset_past_end(self) -> None:
        out = self.tools["read_file"](path="articles/sample.md", offset=len(self.body) + 10)
        self.assertIn("empty slice", out)

    def test_path_escape_rejected(self) -> None:
        out = self.tools["read_file"](path="../../../etc/passwd")
        self.assertIn("escapes workspace", out)


class TestGrepFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        articles = self.workspace / "articles"
        articles.mkdir(parents=True)
        (articles / "hit.md").write_text("line1\nneedle here\nline3\n", encoding="utf-8")
        self.sandbox = AgentSandbox(self.workspace)
        self.fs = FsStore(self.workspace)
        self.episode = AgentEpisodeState()
        _, self.tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid_deep",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_grep_file_only_on_hybrid_deep(self) -> None:
        self.assertIn("grep_file", self.tools)
        _, shell_tools = make_tool_functions(
            sandbox=self.sandbox,
            fs_store=self.fs,
            episode=self.episode,
            toolset="hybrid",
        )
        self.assertNotIn("grep_file", shell_tools)

    def test_grep_finds_pattern(self) -> None:
        out = self.tools["grep_file"](path="articles/hit.md", pattern="needle")
        self.assertIn("needle", out)

    def test_path_escape_rejected(self) -> None:
        out = self.tools["grep_file"](path="../../outside.md", pattern="x")
        self.assertIn("escapes workspace", out)

    def test_multiline_pattern_rejected(self) -> None:
        out = self.tools["grep_file"](path="articles/hit.md", pattern="a\nb")
        self.assertIn("single line", out)


if __name__ == "__main__":
    unittest.main()
