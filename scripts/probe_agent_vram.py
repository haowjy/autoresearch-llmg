#!/usr/bin/env python3
"""Probe agent VRAM / OOM on stress rows (from failed v3 official runs).

Usage:
  uv run python scripts/probe_agent_vram.py
  uv run python scripts/probe_agent_vram.py --rows 5,6,7,8 --out llmg/runs/agent_vram_probe.tsv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

# Stress rows from 20260525-035045 OOM run (long episodes / broad grep).
DEFAULT_STRESS_ROWS = (5, 6, 7, 8)

# Ordered cheapest-first; stop early when a config passes all stress rows.
PROBE_CONFIGS: list[dict[str, str | int | bool]] = [
    {"name": "baseline", "tool_msg_max_chars": 6000, "max_new_tokens": 384},
    {"name": "alloc_expandable", "tool_msg_max_chars": 6000, "max_new_tokens": 384, "alloc_expandable": True},
    {"name": "tool_2000", "tool_msg_max_chars": 2000, "max_new_tokens": 384},
    {"name": "tool_1500", "tool_msg_max_chars": 1500, "max_new_tokens": 384},
    {"name": "tool_1500_t256", "tool_msg_max_chars": 1500, "max_new_tokens": 256},
    {"name": "tool_1200_t256", "tool_msg_max_chars": 1200, "max_new_tokens": 256},
    {
        "name": "tool_1500_t256_expand",
        "tool_msg_max_chars": 1500,
        "max_new_tokens": 256,
        "alloc_expandable": True,
    },
    {
        "name": "tool_1200_t256_expand",
        "tool_msg_max_chars": 1200,
        "max_new_tokens": 256,
        "alloc_expandable": True,
    },
]


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _find_corpus_root() -> Path:
    runs = _repo_root() / "llmg" / "runs"
    latest = runs / "latest" / "corpus_train"
    if latest.is_dir() and (latest / "corpus_manifest.yaml").exists():
        return latest
    for d in sorted(runs.glob("*_P0-TW-03"), reverse=True):
        c = d / "corpus_train"
        if (c / "corpus_manifest.yaml").exists():
            return c
    raise FileNotFoundError("No versioned corpus_train under llmg/runs/")


def _load_stress_rows(row_indices: tuple[int, ...]) -> list[dict]:
    from llmg.data.temporalwiki import load_tw_easy

    ds = load_tw_easy()
    split = ds["test"]
    rows = []
    for i in row_indices:
        rows.append(
            {
                "row_index": i,
                "question": split["question"][i],
                "gold_subject": split["subject_sitelink"][i],
                "gold_answer": split["object"][i],
                "as_of": str(split["snapshot_new"][i] or ""),
            }
        )
    return rows


def _apply_alloc_env(cfg: dict) -> str | None:
    prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if cfg.get("alloc_expandable"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    elif "alloc_expandable" in cfg and not cfg["alloc_expandable"]:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return prev


def _restore_alloc_env(prev: str | None) -> None:
    if prev is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prev


def _probe_config(
    cfg: dict,
    *,
    corpus_root: Path,
    stress_rows: list[dict],
    max_steps: int,
    model_name: str,
) -> dict:
    import torch
    from llmg.agent.gemma_loop import GemmaAgentLoop

    prev_alloc = _apply_alloc_env(cfg)
    t0 = time.perf_counter()
    row_results: list[dict] = []
    oom = False
    error = ""

    agent = GemmaAgentLoop(
        corpus_root=corpus_root,
        model_name=model_name,
        max_steps=max_steps,
        agent_toolset="shell",
        tool_msg_max_chars=int(cfg["tool_msg_max_chars"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        empty_cache_after_episode=True,
    )

    try:
        agent._load_model()
        for row in stress_rows:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            row_t0 = time.perf_counter()
            try:
                result = agent.run_query(
                    row["question"],
                    as_of=row["as_of"],
                    corpus_root=corpus_root,
                    gold_subject=row["gold_subject"],
                    gold_answer=row["gold_answer"],
                    row_index=row["row_index"],
                )
                status = "ok"
                err = ""
                steps = result.steps
            except torch.OutOfMemoryError as exc:
                status = "oom"
                err = str(exc)[:200]
                steps = -1
                oom = True
            except Exception as exc:
                status = "error"
                err = f"{type(exc).__name__}: {exc}"[:200]
                steps = -1

            peak_gb = 0.0
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

            row_results.append(
                {
                    "row_index": row["row_index"],
                    "status": status,
                    "steps": steps,
                    "peak_gb": round(peak_gb, 3),
                    "elapsed_s": round(time.perf_counter() - row_t0, 1),
                    "error": err,
                }
            )
            if oom:
                break
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        _restore_alloc_env(prev_alloc)
        agent._model = None
        agent._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    passed = sum(1 for r in row_results if r["status"] == "ok")
    return {
        "name": cfg["name"],
        "tool_msg_max_chars": cfg["tool_msg_max_chars"],
        "max_new_tokens": cfg["max_new_tokens"],
        "alloc_expandable": bool(cfg.get("alloc_expandable")),
        "passed_rows": passed,
        "total_rows": len(stress_rows),
        "all_ok": passed == len(stress_rows) and not oom,
        "oom": oom,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "rows": row_results,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe agent VRAM on stress rows")
    parser.add_argument(
        "--rows",
        default=",".join(str(x) for x in DEFAULT_STRESS_ROWS),
        help="Comma-separated test split row indices",
    )
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "llmg" / "runs" / "agent_vram_probe.json",
    )
    parser.add_argument("--stop-on-first-pass", action="store_true")
    args = parser.parse_args()

    row_indices = tuple(int(x.strip()) for x in args.rows.split(",") if x.strip())
    corpus_root = _find_corpus_root()
    stress_rows = _load_stress_rows(row_indices)

    print(f"corpus: {corpus_root}")
    print(f"stress rows: {row_indices}")
    print(f"max_steps: {args.max_steps}")
    print(f"configs: {len(PROBE_CONFIGS)}")
    sys.stdout.flush()

    results: list[dict] = []
    winner: dict | None = None

    for cfg in PROBE_CONFIGS:
        print(f"\n=== {cfg['name']} ===", flush=True)
        summary = _probe_config(
            cfg,
            corpus_root=corpus_root,
            stress_rows=stress_rows,
            max_steps=args.max_steps,
            model_name=args.model,
        )
        results.append(summary)
        for r in summary["rows"]:
            print(
                f"  row {r['row_index']}: {r['status']} steps={r['steps']} "
                f"peak={r['peak_gb']}GB {r['elapsed_s']}s",
                flush=True,
            )
        print(
            f"  => {summary['passed_rows']}/{summary['total_rows']} ok, "
            f"total {summary['elapsed_s']}s",
            flush=True,
        )
        if summary["all_ok"] and winner is None:
            winner = summary
            print(f"  ** first all-pass: {cfg['name']} **", flush=True)
            if args.stop_on_first_pass:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_root": str(corpus_root),
        "stress_rows": list(row_indices),
        "max_steps": args.max_steps,
        "results": results,
        "winner": winner["name"] if winner else None,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tsv_path = args.out.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "config",
                "tool_msg_max",
                "max_new_tokens",
                "alloc_expandable",
                "passed",
                "total",
                "all_ok",
                "elapsed_s",
            ]
        )
        for s in results:
            w.writerow(
                [
                    s["name"],
                    s["tool_msg_max_chars"],
                    s["max_new_tokens"],
                    int(s["alloc_expandable"]),
                    s["passed_rows"],
                    s["total_rows"],
                    int(s["all_ok"]),
                    s["elapsed_s"],
                ]
            )

    print(f"\nWrote {args.out} and {tsv_path}")
    if winner:
        print(
            f"Recommended env for official v3:\n"
            f"  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # if winner uses it\n"
            f"  export LLMG_AGENT_TOOL_MSG_MAX={winner['tool_msg_max_chars']}\n"
            f"  export LLMG_AGENT_MAX_NEW_TOKENS={winner['max_new_tokens']}"
        )
        return 0
    print("No config passed all stress rows.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
