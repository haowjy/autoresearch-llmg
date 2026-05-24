#!/usr/bin/env python3
"""Per-row answer quality from P0-TW-03 agent_traces (episode_start / episode_end)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmg.eval.temporal_metrics import (
    ANSWER_COSINE_HIT_THRESHOLD,
    answer_cosine_similarity,
    answer_cosine_similarity_batch,
    answer_exact_match,
    subject_recall_at_k,
    temporal_recall_at_k,
)
from llmg.memory.doc_catalog import DocCatalog


def _load_rows(trace_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(trace_dir.glob("row_*.jsonl"), key=lambda x: int(x.stem.split("_")[1])):
        row = {"row": int(p.stem.split("_")[1]), "gold_sub": "", "gold_ans": "", "pred": "", "as_of": "", "retrieved": []}
        for line in p.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("type") == "episode_start":
                row["gold_sub"] = ev.get("gold_subject", "")
                row["gold_ans"] = ev.get("gold_answer", "")
                row["as_of"] = ev.get("as_of", "")
            elif ev.get("type") == "episode_end":
                row["pred"] = ev.get("answer", "")
                row["retrieved"] = ev.get("retrieved_doc_ids", [])
        if row["gold_ans"]:
            rows.append(row)
    return rows


def summarize(rows: list[dict], doc_meta: dict[str, tuple[str, str, str]], *, k: int = 5) -> dict:
    n = len(rows)
    cos_all = answer_cosine_similarity_batch([(r["pred"], r["gold_ans"]) for r in rows])
    em = [answer_exact_match(r["pred"], r["gold_ans"]) for r in rows]
    answered = [r for r in rows if r["pred"].strip()]
    cos_ans = [answer_cosine_similarity(r["pred"], r["gold_ans"]) for r in answered]

    sub_r, temp_r = [], []
    for r in rows:
        subs = [doc_meta[d][0] for d in r["retrieved"][:k] if d in doc_meta]
        sub_r.append(subject_recall_at_k(subs, r["gold_sub"], k))
        temp_r.append(
            temporal_recall_at_k(
                r["retrieved"],
                gold_subject=r["gold_sub"],
                as_of=r["as_of"],
                doc_meta=doc_meta,
                k=k,
            )
        )

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    sub_miss = sum(1 for x in sub_r if not x)
    sub_ok_temp_miss = sum(1 for sr, tr in zip(sub_r, temp_r) if sr and not tr)

    return {
        "n": n,
        "no_submit": n - len(answered),
        "em_rate_all": _mean(em),
        "cos_mean_all": _mean(cos_all),
        "cos_hit_rate_all": sum(1 for c in cos_all if c >= ANSWER_COSINE_HIT_THRESHOLD) / n,
        "answered_n": len(answered),
        "em_rate_answered": sum(answer_exact_match(r["pred"], r["gold_ans"]) for r in answered) / len(answered),
        "cos_mean_answered": _mean(cos_ans),
        "cos_hit_rate_answered": sum(1 for c in cos_ans if c >= ANSWER_COSINE_HIT_THRESHOLD) / len(answered),
        "subject_recall@k": _mean([float(x) for x in sub_r]),
        "temporal_recall@k": _mean([float(x) for x in temp_r]),
        "subject_miss": sub_miss,
        "subject_ok_temporal_miss": sub_ok_temp_miss,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="e.g. llmg/runs/20260524-023355_P0-TW-03")
    ap.add_argument("--cell", required=True, help="agent_traces subdir name")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    corpus = args.run_dir / "corpus_train"
    if not corpus.is_dir():
        corpus = args.run_dir / "corpus_train_stable"
    doc_meta = DocCatalog.from_corpus_root(corpus).doc_meta_tuple()
    trace_dir = args.run_dir / "agent_traces" / args.cell
    stats = summarize(_load_rows(trace_dir), doc_meta, k=args.k)
    print(json.dumps({"cell": args.cell, **stats}, indent=2))


if __name__ == "__main__":
    main()
