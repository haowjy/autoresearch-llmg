#!/usr/bin/env python3
"""Audit whether search_hybrid snippets expose relation-specific answer candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmg.agent.search_candidates import candidate_matches, extract_search_candidates
from llmg.eval.temporal_metrics import answer_exact_match


def _norm(text: str) -> str:
    import re

    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s]", "", text)


def _load_rows(trace_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(trace_dir.glob("row_*.jsonl"), key=lambda x: int(x.stem.split("_")[1])):
        row = {
            "row": int(p.stem.split("_")[1]),
            "row_index": int(p.stem.split("_")[1]),
            "question": "",
            "gold_subject": "",
            "gold_answer": "",
            "prediction": "",
            "answer_source": "none",
            "search_text": "",
        }
        for line in p.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("type") == "episode_start":
                row["question"] = str(ev.get("question") or "")
                row["gold_subject"] = str(ev.get("gold_subject") or "")
                row["gold_answer"] = str(ev.get("gold_answer") or "")
                row["row_index"] = int(ev.get("row_index", row["row_index"]))
            elif ev.get("type") == "tool_result" and ev.get("name") == "search_hybrid":
                row["search_text"] += "\n" + str(ev.get("content") or "")
            elif ev.get("type") == "episode_end":
                row["prediction"] = str(ev.get("answer") or "")
                row["answer_source"] = str(ev.get("answer_source") or "none")
        if row["gold_answer"]:
            rows.append(row)
    return rows


def _find_trace_cell(run_dir: Path, requested: str | None) -> tuple[str, Path]:
    traces = run_dir / "agent_traces"
    if requested:
        return requested, traces / requested
    cells = sorted([p for p in traces.iterdir() if p.is_dir()]) if traces.is_dir() else []
    if len(cells) != 1:
        names = ", ".join(p.name for p in cells) or "(none)"
        raise SystemExit(f"expected exactly one trace cell; pass --cell. Found: {names}")
    return cells[0].name, cells[0]


def summarize(rows: list[dict], *, include_rows: bool) -> dict:
    details = []
    candidate_rows = 0
    top1_hits = 0
    any_hits = 0
    pred_in_candidates = 0
    gold_visible = 0
    for row in rows:
        candidates = extract_search_candidates(row["question"], row["search_text"])
        if candidates:
            candidate_rows += 1
        top1_hit = bool(candidates and answer_exact_match(candidates[0], row["gold_answer"]))
        any_hit = any(answer_exact_match(c, row["gold_answer"]) for c in candidates)
        pred_hit = any(candidate_matches(c, row["prediction"]) for c in candidates)
        gold_is_visible = _norm(row["gold_answer"]) in _norm(row["search_text"])
        top1_hits += int(top1_hit)
        any_hits += int(any_hit)
        pred_in_candidates += int(pred_hit)
        gold_visible += int(gold_is_visible)
        if include_rows:
            details.append(
                {
                    "row": row["row"],
                    "row_index": row["row_index"],
                    "gold_answer": row["gold_answer"],
                    "prediction": row["prediction"],
                    "answer_source": row["answer_source"],
                    "gold_visible_in_search": gold_is_visible,
                    "candidate_top1": candidates[0] if candidates else "",
                    "candidate_top1_exact": top1_hit,
                    "candidate_any_exact": any_hit,
                    "prediction_in_candidates": pred_hit,
                    "candidates": candidates[:8],
                }
            )
    n = len(rows)
    out = {
        "n": n,
        "gold_visible_in_search": gold_visible,
        "candidate_rows": candidate_rows,
        "candidate_top1_em": top1_hits / n if n else 0.0,
        "candidate_any_em": any_hits / n if n else 0.0,
        "prediction_in_candidates": pred_in_candidates,
    }
    if include_rows:
        out["rows"] = details
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--cell")
    ap.add_argument("--rows", action="store_true")
    args = ap.parse_args()

    cell, trace_dir = _find_trace_cell(args.run_dir, args.cell)
    stats = summarize(_load_rows(trace_dir), include_rows=args.rows)
    print(json.dumps({"cell": cell, **stats}, indent=2))


if __name__ == "__main__":
    main()
