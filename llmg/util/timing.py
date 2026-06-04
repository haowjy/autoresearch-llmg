"""Wall-clock phase timing for experiment runs (perf_counter-based)."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

TIMING_FILENAME = "timing.json"
PARTIAL_FILENAME = "timing_partial.json"


def gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 4)
    except Exception:
        pass
    return None


class PhaseTimer:
    """Track named phases with time.perf_counter()."""

    def __init__(self, *, record_gpu: bool = False) -> None:
        self._record_gpu = record_gpu
        self._starts: dict[str, float] = {}
        self._phases_s: dict[str, float] = {}
        self._gpu_gb: dict[str, float | None] = {}

    def start(self, name: str) -> None:
        if name in self._starts:
            raise ValueError(f"phase already active: {name!r}")
        self._starts[name] = time.perf_counter()
        if self._record_gpu:
            self._gpu_gb[f"{name}_start"] = gpu_memory_gb()

    def stop(self, name: str) -> float:
        t0 = self._starts.pop(name, None)
        if t0 is None:
            raise ValueError(f"phase not started: {name!r}")
        elapsed = time.perf_counter() - t0
        self._phases_s[name] = round(elapsed, 4)
        if self._record_gpu:
            self._gpu_gb[f"{name}_end"] = gpu_memory_gb()
        return elapsed

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def as_dict(self) -> dict[str, float]:
        return dict(self._phases_s)

    def as_report(self, *, experiment_wall_s: float | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {"phases_s": dict(self._phases_s)}
        if experiment_wall_s is not None:
            report["experiment_wall_s"] = round(experiment_wall_s, 4)
        elif self._phases_s:
            report["experiment_wall_s"] = round(max(self._phases_s.values()), 4)
        if self._gpu_gb:
            report["gpu_memory_gb"] = self._gpu_gb
        return report


def write_timing_json(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_timing_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_partial_timing(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Merge timing_partial.json from subprocesses into *report* (in place)."""
    partial_path = run_dir / PARTIAL_FILENAME
    if not partial_path.is_file():
        return report
    partial = read_timing_json(partial_path)
    phases = report.setdefault("phases_s", {})
    for name, seconds in (partial.get("phases_s") or {}).items():
        phases.setdefault(name, seconds)
    gpu = partial.get("gpu_memory_gb")
    if gpu:
        merged = report.setdefault("gpu_memory_gb", {})
        merged.update(gpu)
    return report


def record_partial_phase(
    run_dir: Path,
    name: str,
    seconds: float,
    *,
    record_gpu: bool = False,
) -> None:
    """Append one phase from a subprocess (atomic read-merge-write)."""
    path = run_dir / PARTIAL_FILENAME
    data = read_timing_json(path) if path.is_file() else {}
    phases = data.setdefault("phases_s", {})
    phases[name] = round(seconds, 4)
    if record_gpu:
        snap = gpu_memory_gb()
        if snap is not None:
            gpu = data.setdefault("gpu_memory_gb", {})
            gpu[f"{name}_end"] = snap
    write_timing_json(path, data)


def finalize_timing(run_dir: Path, report: dict[str, Any]) -> Path:
    """Merge partials and write run_dir/timing.json."""
    merged = merge_partial_timing(run_dir, report)
    out = run_dir / TIMING_FILENAME
    write_timing_json(out, merged)
    return out


def timing_metrics_flat(report: dict[str, Any]) -> dict[str, float]:
    """Flatten phases_s and experiment_wall_s for metrics.json / results.tsv."""
    out: dict[str, float] = {}
    wall = report.get("experiment_wall_s")
    if wall is not None:
        out["experiment_wall_s"] = float(wall)
    for name, seconds in (report.get("phases_s") or {}).items():
        out[f"{name}_s"] = float(seconds)
    return out
