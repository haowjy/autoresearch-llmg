"""Subprocess entry: QLoRA training only (keeps parent llmg.run off the GPU)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from llmg.train.qlora import train_naive_qlora
from llmg.util.timing import PhaseTimer, record_partial_phase

log = logging.getLogger(__name__)


def train_from_run_dir(run_dir: Path) -> Path:
    cfg = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
    params = cfg.get("params") or {}
    phase = cfg.get("run_phase", "official")

    max_train_steps = params.get("max_train_steps")
    max_train_rows = params.get("max_train_rows")
    if phase == "calibrate":
        max_train_steps = params.get("calibrate_max_train_steps", 30)
        max_train_rows = params.get("calibrate_max_train_rows", 128)

    from llmg.data.temporalwiki import load_tw_easy

    ds = load_tw_easy()
    timer = PhaseTimer(record_gpu=True)
    timer.start("qlora_train")
    try:
        adapter = train_naive_qlora(
            train_split=ds["train"],
            output_dir=run_dir,
            base_model=str(params.get("base_model", "google/gemma-4-E4B-it")),
            lora_rank=int(params.get("lora_rank", 8)),
            lora_alpha=int(params.get("lora_alpha", 16)),
            learning_rate=float(params.get("learning_rate", 2e-4)),
            train_batch_size=int(params.get("train_batch_size", 1)),
            gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 4)),
            max_seq_len=int(params.get("max_seq_len", 2048)),
            train_epochs=float(params.get("train_epochs", 1)),
            max_train_steps=int(max_train_steps) if max_train_steps is not None else None,
            max_train_rows=int(max_train_rows) if max_train_rows is not None else None,
            max_article_chars=int(params.get("max_article_chars", 1500)),
            sft_format=str(params.get("sft_format", "plain")),
        )
    finally:
        seconds = timer.stop("qlora_train")
        record_partial_phase(run_dir, "qlora_train", seconds, record_gpu=True)
    return adapter


def main() -> None:
    from llmg.util.hf_local import configure_hf_offline_if_requested

    configure_hf_offline_if_requested()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    adapter = train_from_run_dir(args.run_dir)
    print(adapter)


if __name__ == "__main__":
    main()
