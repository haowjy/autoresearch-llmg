#!/usr/bin/env python3
"""Per-row answer quality from agent traces (episode_start / episode_end)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        row = {
            "row": int(p.stem.split("_")[1]),
            "row_index": int(p.stem.split("_")[1]),
            "gold_sub": "",
            "gold_ans": "",
            "pred": "",
            "as_of": "",
            "retrieved": [],
            "answer_source": "none",
            "invalid_submit_count": 0,
            "tool_text": "",
            "search_text": "",
        }
        for line in p.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev.get("type") == "episode_start":
                row["gold_sub"] = ev.get("gold_subject", "")
                row["gold_ans"] = ev.get("gold_answer", "")
                row["as_of"] = ev.get("as_of", "")
                row["row_index"] = int(ev.get("row_index", row["row_index"]))
            elif ev.get("type") == "episode_end":
                row["pred"] = ev.get("answer", "")
                row["retrieved"] = ev.get("retrieved_doc_ids", [])
                row["answer_source"] = ev.get("answer_source", "none")
                row["invalid_submit_count"] = int(ev.get("invalid_submit_count") or 0)
            elif ev.get("type") == "tool_result":
                content = str(ev.get("content") or "")
                row["tool_text"] = f"{row['tool_text']}\n{content}"
                if ev.get("name") == "search_hybrid":
                    row["search_text"] = f"{row['search_text']}\n{content}"
        if row["gold_ans"]:
            rows.append(row)
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _norm_text(text: str) -> str:
    import re

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def _retrieved_subjects(row: dict, doc_meta: dict[str, tuple[str, str, str]], *, k: int) -> list[str]:
    return [doc_meta[d][0] for d in row["retrieved"][:k] if d in doc_meta]


def _co_mentioned_in_gold_doc(row: dict, corpus_root: Path) -> bool:
    pred = _norm_text(row["pred"])
    gold = _norm_text(row["gold_ans"])
    if not pred or not gold:
        return False
    for doc_id in row["retrieved"]:
        path = corpus_root / "articles" / f"{doc_id}.md"
        if not path.is_file():
            continue
        text = _norm_text(path.read_text(encoding="utf-8"))
        if pred in text and gold in text:
            return True
    return False


def _contains_norm(haystack: str, needle: str) -> bool:
    norm_needle = _norm_text(needle)
    return bool(norm_needle and norm_needle in _norm_text(haystack))


def _alias_linked_in_gold_doc(row: dict, corpus_root: Path) -> bool:
    import re

    pred = _norm_text(row["pred"])
    gold = _norm_text(row["gold_ans"])
    if not pred or not gold:
        return False
    alias_markers = (
        "also known as",
        "alternately",
        "alternatively",
        "westernized as",
        "romanized as",
        "stylized as",
        "styled as",
        "born",
        "known professionally as",
        "alias",
    )
    for doc_id in row["retrieved"]:
        path = corpus_root / "articles" / f"{doc_id}.md"
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            norm_sentence = _norm_text(sentence)
            if pred not in norm_sentence or gold not in norm_sentence:
                continue
            if any(marker in norm_sentence for marker in alias_markers):
                return True
    return False


def classify_row(
    row: dict,
    doc_meta: dict[str, tuple[str, str, str]],
    corpus_root: Path,
    *,
    k: int = 5,
    cosine: float = 0.0,
) -> str:
    subjects = _retrieved_subjects(row, doc_meta, k=k)
    subject_hit = subject_recall_at_k(subjects, row["gold_sub"], k)
    temporal_hit = temporal_recall_at_k(
        row["retrieved"],
        gold_subject=row["gold_sub"],
        as_of=row["as_of"],
        doc_meta=doc_meta,
        k=k,
    )
    if answer_exact_match(row["pred"], row["gold_ans"]):
        return "exact"
    if not row["pred"].strip():
        return "no_answer"
    if not subject_hit:
        return "subject_retrieval_miss"
    if not temporal_hit:
        return "temporal_retrieval_miss"
    if row.get("answer_source") == "last_content_fallback":
        return "fallback_wrong"
    if cosine >= ANSWER_COSINE_HIT_THRESHOLD:
        return "semantic_close_wrong_surface"
    pred_norm = _norm_text(row["pred"])
    gold_norm = _norm_text(row["gold_ans"])
    if pred_norm and gold_norm and (pred_norm in gold_norm or gold_norm in pred_norm):
        return "contains_wrong_surface"
    if _alias_linked_in_gold_doc(row, corpus_root):
        return "alias_linked_wrong_surface"
    if _co_mentioned_in_gold_doc(row, corpus_root):
        return "gold_doc_co_mentions_pred"
    return "wrong_extraction"


def summarize(
    rows: list[dict],
    doc_meta: dict[str, tuple[str, str, str]],
    corpus_root: Path,
    *,
    k: int = 5,
    include_rows: bool = False,
) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "buckets": {}, "rows": [] if include_rows else None}
    cos_all = answer_cosine_similarity_batch([(r["pred"], r["gold_ans"]) for r in rows])
    em = [answer_exact_match(r["pred"], r["gold_ans"]) for r in rows]
    answered = [r for r in rows if r["pred"].strip()]
    cos_ans = [answer_cosine_similarity(r["pred"], r["gold_ans"]) for r in answered]

    sub_r, temp_r = [], []
    row_details: list[dict] = []
    buckets: dict[str, int] = {}
    gold_in_tool = 0
    pred_in_tool = 0
    gold_in_search = 0
    pred_in_search = 0
    for i, r in enumerate(rows):
        subs = _retrieved_subjects(r, doc_meta, k=k)
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
        bucket = classify_row(r, doc_meta, corpus_root, k=k, cosine=cos_all[i])
        buckets[bucket] = buckets.get(bucket, 0) + 1
        row_gold_in_tool = _contains_norm(r.get("tool_text", ""), r["gold_ans"])
        row_pred_in_tool = _contains_norm(r.get("tool_text", ""), r["pred"])
        row_gold_in_search = _contains_norm(r.get("search_text", ""), r["gold_ans"])
        row_pred_in_search = _contains_norm(r.get("search_text", ""), r["pred"])
        gold_in_tool += int(row_gold_in_tool)
        pred_in_tool += int(row_pred_in_tool)
        gold_in_search += int(row_gold_in_search)
        pred_in_search += int(row_pred_in_search)
        if include_rows:
            row_details.append(
                {
                    "row": r["row"],
                    "row_index": r["row_index"],
                    "bucket": bucket,
                    "gold_subject": r["gold_sub"],
                    "gold_answer": r["gold_ans"],
                    "prediction": r["pred"],
                    "answer_source": r["answer_source"],
                    "cosine": cos_all[i],
                    "subject_hit": sub_r[-1],
                    "temporal_hit": temp_r[-1],
                    "gold_in_tool_text": row_gold_in_tool,
                    "pred_in_tool_text": row_pred_in_tool,
                    "gold_in_search_text": row_gold_in_search,
                    "pred_in_search_text": row_pred_in_search,
                    "retrieved": r["retrieved"][:k],
                }
            )

    sub_miss = sum(1 for x in sub_r if not x)
    sub_ok_temp_miss = sum(1 for sr, tr in zip(sub_r, temp_r) if sr and not tr)

    out = {
        "n": n,
        "no_submit": n - len(answered),
        "em_rate_all": _mean(em),
        "cos_mean_all": _mean(cos_all),
        "cos_hit_rate_all": sum(1 for c in cos_all if c >= ANSWER_COSINE_HIT_THRESHOLD) / n,
        "answered_n": len(answered),
        "em_rate_answered": (
            sum(answer_exact_match(r["pred"], r["gold_ans"]) for r in answered) / len(answered)
            if answered
            else 0.0
        ),
        "cos_mean_answered": _mean(cos_ans),
        "cos_hit_rate_answered": (
            sum(1 for c in cos_ans if c >= ANSWER_COSINE_HIT_THRESHOLD) / len(answered)
            if answered
            else 0.0
        ),
        "subject_recall@k": _mean([float(x) for x in sub_r]),
        "temporal_recall@k": _mean([float(x) for x in temp_r]),
        "subject_miss": sub_miss,
        "subject_ok_temporal_miss": sub_ok_temp_miss,
        "gold_in_tool_text": gold_in_tool,
        "pred_in_tool_text": pred_in_tool,
        "gold_in_search_text": gold_in_search,
        "pred_in_search_text": pred_in_search,
        "buckets": buckets,
    }
    if include_rows:
        out["rows"] = row_details
    return out


def _find_trace_cell(run_dir: Path, requested: str | None) -> tuple[str, Path]:
    traces = run_dir / "agent_traces"
    if requested:
        return requested, traces / requested
    cells = sorted([p for p in traces.iterdir() if p.is_dir()]) if traces.is_dir() else []
    if len(cells) != 1:
        names = ", ".join(p.name for p in cells) or "(none)"
        raise SystemExit(f"expected exactly one trace cell; pass --cell. Found: {names}")
    return cells[0].name, cells[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path, help="e.g. llmg/runs/20260524-023355_P0-TW-03")
    ap.add_argument("--cell", help="agent_traces subdir name; auto-detected when only one exists")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--rows", action="store_true", help="include per-row bucket details")
    args = ap.parse_args()
    corpus = args.run_dir / "corpus_train"
    if not corpus.is_dir():
        corpus = args.run_dir / "corpus_train_stable"
    doc_meta = DocCatalog.from_corpus_root(corpus).doc_meta_tuple()
    cell, trace_dir = _find_trace_cell(args.run_dir, args.cell)
    stats = summarize(_load_rows(trace_dir), doc_meta, corpus, k=args.k, include_rows=args.rows)
    print(json.dumps({"cell": cell, **stats}, indent=2))


if __name__ == "__main__":
    main()
