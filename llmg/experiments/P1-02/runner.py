"""P1-02 — naive QLoRA on TemporalWiki train Q/A, then hybrid-agent RAG eval."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from llmg.util.timing import (
    PhaseTimer,
    finalize_timing,
    read_timing_json,
    timing_metrics_flat,
)

if TYPE_CHECKING:
    from llmg.observability.run_session import RunSession

log = logging.getLogger(__name__)


def run(
    *,
    k: int = 5,
    eval_split: str = "test",
    base_model: str = "google/gemma-4-E4B-it",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    train_epochs: float = 1.0,
    max_train_steps: int | None = None,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
    eval_start_row: int = 0,
    learning_rate: float = 2e-4,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_len: int = 2048,
    max_article_chars: int = 1500,
    sft_format: str = "plain",
    agent_toolset: str = "hybrid",
    corpus_index_mode: str = "train_stable",
    max_agent_steps: int = 16,
    search_snippets_per_hit: int | None = None,
    search_snippet_hits: int | None = None,
    eval_relation_hint: bool | str = False,
    adapter_source_run: str | None = None,
    skip_train: bool = False,
    skip_eval: bool = False,
    calibrate_max_train_steps: int = 30,
    calibrate_max_train_rows: int = 1500,
    calibrate_max_eval_rows: int = 10,
    session: RunSession | None = None,
) -> dict[str, float]:
    if session is None:
        raise ValueError("P1-02 requires a RunSession (use llmg.run)")

    run_dir = session.run_dir
    phase = session.run_phase
    _ = (
        k,
        eval_split,
        base_model,
        lora_rank,
        lora_alpha,
        train_epochs,
        max_train_steps,
        max_train_rows,
        max_eval_rows,
        eval_start_row,
        learning_rate,
        train_batch_size,
        gradient_accumulation_steps,
        max_seq_len,
        max_article_chars,
        sft_format,
        agent_toolset,
        max_agent_steps,
        search_snippets_per_hit,
        search_snippet_hits,
        eval_relation_hint,
        adapter_source_run,
    )

    timer = PhaseTimer()
    timer.start("run")
    effective_train_steps = max_train_steps
    effective_train_rows = max_train_rows
    if phase == "calibrate":
        effective_train_steps = calibrate_max_train_steps
        effective_train_rows = calibrate_max_train_rows

    metrics: dict[str, float] = {
        "train_steps": float(effective_train_steps or -1),
    }

    if not skip_train:
        session.log("spawning QLoRA train subprocess (phase=%s)", phase)
        with timer.phase("qlora_train"):
            subprocess.run(
                [sys.executable, "-m", "llmg.train.run_qlora_train", str(run_dir)],
                check=True,
            )
        metrics["train_rows"] = float(effective_train_rows or 1500)

    adapter_dir = run_dir / "lora_adapter"
    if adapter_source_run:
        source = Path(str(adapter_source_run))
        if not source.is_absolute():
            source = Path("llmg/runs") / source
        adapter_dir = source if source.name == "lora_adapter" else source / "lora_adapter"
    if not skip_eval and not adapter_dir.is_dir():
        timer.stop("run")
        raise FileNotFoundError(f"no adapter at {adapter_dir}; run training first.")

    if skip_eval:
        return _finish_timing(session, run_dir, timer, metrics)

    # Corpus export + dataset load happen in the eval subprocess only (avoids a third Hub pass).
    session.log("spawning LoRA eval subprocess")
    with timer.phase("lora_eval"):
        subprocess.run(
            [sys.executable, "-m", "llmg.train.run_lora_eval", str(run_dir)],
            check=True,
        )

    eval_path = run_dir / "lora_eval_metrics.json"
    if eval_path.is_file():
        metrics.update(json.loads(eval_path.read_text(encoding="utf-8")))
    return _finish_timing(session, run_dir, timer, metrics)


def _finish_timing(
    session: RunSession,
    run_dir,
    timer: PhaseTimer,
    metrics: dict[str, float],
) -> dict[str, float]:
    timer.stop("run")
    phases = timer.as_dict()
    path = finalize_timing(
        run_dir,
        timer.as_report(experiment_wall_s=phases.get("run")),
    )
    flat = timing_metrics_flat(read_timing_json(path))
    metrics.update(flat)
    session.log(
        "timing: qlora_train=%.1fs lora_eval=%.1fs run=%.1fs",
        flat.get("qlora_train_s", 0.0),
        flat.get("lora_eval_s", 0.0),
        flat.get("run_s", 0.0),
    )
    return metrics
