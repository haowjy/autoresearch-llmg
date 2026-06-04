"""Per-run artifact directory: logs, config, metrics next to each experiment run."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"
RESULTS_TSV = Path(__file__).resolve().parents[2] / "results.tsv"

TSV_HEADER = (
    "commit\trun_phase\texperiment_id\tprimary_metric\tscore\t"
    "memory_gb\tstatus\tdescription\trun_dir\n"
)


@dataclass
class RunSession:
    experiment_id: str
    run_phase: str = "official"
    description: str = ""
    run_dir: Path = field(init=False)
    primary_metric: str = ""
    score: float | None = None
    status: str = "ok"
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_id = self.experiment_id.replace("/", "-")
        self.run_dir = RUNS_ROOT / f"{stamp}_{safe_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._logger = _configure_run_logging(self.run_dir)
        self._write_config()
        self._link_latest()
        self.log(f"run_dir={self.run_dir}")

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def log(self, msg: str, *args: Any, level: int = logging.INFO) -> None:
        self._logger.log(level, msg, *args)

    def finish(
        self,
        metrics: dict[str, float],
        *,
        primary_metric: str | None = None,
        status: str | None = None,
        description: str | None = None,
    ) -> Path:
        if status is not None:
            self.status = status
        if description is not None:
            self.description = description

        primary = primary_metric or self.primary_metric
        if not primary:
            for key in metrics:
                if key.startswith("retrieval_recall"):
                    primary = key
                    break
            else:
                primary = next(iter(metrics), "score")

        self.primary_metric = primary
        self.score = float(metrics.get(primary, metrics.get("score", 0.0)))

        elapsed = time.perf_counter() - self._t0
        meta = _gather_meta(elapsed)
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = _format_summary(self, metrics, meta)
        (self.run_dir / "summary.md").write_text(summary, encoding="utf-8")

        self.log(f"primary_metric={self.primary_metric} score={self.score:.4f} status={self.status}")
        self.log(f"elapsed_s={elapsed:.1f}")

        _print_score_lines(self)
        _append_results_tsv(self)
        return self.run_dir

    def fail(self, exc: BaseException) -> Path:
        self.status = "crash"
        self.score = None
        self.log(f"crash: {exc!r}", level=logging.ERROR)
        import traceback

        (self.run_dir / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        elapsed = time.perf_counter() - self._t0
        meta = _gather_meta(elapsed)
        meta["error"] = repr(exc)
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _append_results_tsv(self)
        return self.run_dir

    def _write_config(self) -> None:
        cfg = {
            "experiment_id": self.experiment_id,
            "run_phase": self.run_phase,
            "description": self.description,
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "git_commit": _git_commit(),
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _link_latest(self) -> None:
        latest = RUNS_ROOT / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.run_dir.name, target_is_directory=True)
        except OSError:
            # Windows or FS without symlinks: write pointer file
            latest.write_text(self.run_dir.name + "\n", encoding="utf-8")


def _configure_run_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("llmg.run")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _gpu_memory_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return 0.0


def _gather_meta(elapsed_s: float) -> dict[str, Any]:
    wall = round(elapsed_s, 4)
    return {
        "elapsed_s": wall,
        "experiment_wall_s": wall,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "memory_gb": _gpu_memory_gb(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def _format_summary(session: RunSession, metrics: dict[str, float], meta: dict[str, Any]) -> str:
    lines = [
        f"# Run {session.experiment_id}",
        "",
        f"- **run_dir:** `{session.run_dir}`",
        f"- **phase:** {session.run_phase}",
        f"- **status:** {session.status}",
        f"- **elapsed:** {meta.get('experiment_wall_s', meta.get('elapsed_s'))}s",
        "",
        "## Primary",
        "",
        f"- **{session.primary_metric}:** {session.score}",
        "",
        "## All metrics",
        "",
    ]
    for k, v in sorted(metrics.items()):
        lines.append(f"- `{k}`: {v}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `run.log` — full text log",
            "- `config.json` — argv + git commit",
            "- `metrics.json` — machine-readable scores",
            "- `meta.json` — host + experiment_wall_s",
            "- `timing.json` — per-phase wall times (when recorded)",
            "",
        ]
    )
    return "\n".join(lines)


def _print_score_lines(session: RunSession) -> None:
    print(f"run_dir: {session.run_dir}")
    print(f"experiment_id: {session.experiment_id}")
    print(f"primary_metric: {session.primary_metric}")
    if session.score is not None:
        print(f"score: {session.score:.4f}")
    print(f"status: {session.status}")


def _append_results_tsv(session: RunSession) -> None:
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(TSV_HEADER, encoding="utf-8")
    commit = _git_commit()
    score_str = "" if session.score is None else f"{session.score:.6f}"
    mem = _gpu_memory_gb()
    run_dir_rel = session.run_dir.relative_to(RESULTS_TSV.parent)
    row = (
        f"{commit}\t{session.run_phase}\t{session.experiment_id}\t"
        f"{session.primary_metric}\t{score_str}\t{mem}\t{session.status}\t"
        f"{session.description}\t{run_dir_rel}\n"
    )
    with RESULTS_TSV.open("a", encoding="utf-8") as f:
        f.write(row)

