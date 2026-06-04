"""Unit tests for llmg.util.timing (no GPU)."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from llmg.util.timing import (
    PhaseTimer,
    finalize_timing,
    merge_partial_timing,
    read_timing_json,
    record_partial_phase,
    timing_metrics_flat,
    write_timing_json,
)


class TestPhaseTimer(unittest.TestCase):
    def test_start_stop_as_dict(self) -> None:
        timer = PhaseTimer()
        timer.start("a")
        time.sleep(0.01)
        timer.stop("a")
        d = timer.as_dict()
        self.assertIn("a", d)
        self.assertGreaterEqual(d["a"], 0.01)

    def test_phase_context_manager(self) -> None:
        timer = PhaseTimer()
        with timer.phase("b"):
            time.sleep(0.005)
        self.assertIn("b", timer.as_dict())

    def test_double_start_raises(self) -> None:
        timer = PhaseTimer()
        timer.start("x")
        with self.assertRaises(ValueError):
            timer.start("x")
        timer.stop("x")

    def test_stop_without_start_raises(self) -> None:
        timer = PhaseTimer()
        with self.assertRaises(ValueError):
            timer.stop("missing")

    def test_as_report_includes_wall(self) -> None:
        timer = PhaseTimer()
        with timer.phase("run"):
            pass
        report = timer.as_report(experiment_wall_s=1.5)
        self.assertEqual(report["experiment_wall_s"], 1.5)
        self.assertIn("run", report["phases_s"])


class TestTimingArtifacts(unittest.TestCase):
    def test_merge_partial_and_finalize(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            record_partial_phase(run_dir, "qlora_train", 12.3)
            timer = PhaseTimer()
            with timer.phase("lora_eval"):
                pass
            report = timer.as_report(experiment_wall_s=20.0)
            path = finalize_timing(run_dir, report)
            self.assertTrue(path.is_file())
            data = read_timing_json(path)
            self.assertEqual(data["phases_s"]["qlora_train"], 12.3)
            self.assertIn("lora_eval", data["phases_s"])

    def test_timing_metrics_flat(self) -> None:
        flat = timing_metrics_flat(
            {"experiment_wall_s": 10.0, "phases_s": {"qlora_train": 3.0}}
        )
        self.assertEqual(flat["experiment_wall_s"], 10.0)
        self.assertEqual(flat["qlora_train_s"], 3.0)

    def test_write_read_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "timing.json"
            write_timing_json(path, {"phases_s": {"run": 1.0}})
            self.assertEqual(read_timing_json(path)["phases_s"]["run"], 1.0)

    def test_merge_partial_no_file(self) -> None:
        report: dict = {"phases_s": {"run": 1.0}}
        merge_partial_timing(Path("/nonexistent"), report)
        self.assertEqual(report["phases_s"]["run"], 1.0)


if __name__ == "__main__":
    unittest.main()
