"""P0-TW-05 — BM25-hard subset on TemporalWiki drift CL base."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmg.agent.gemma_loop import run_agent_eval
from llmg.data.corpus_export import (
    corpus_is_current,
    dedupe_records,
    export_all,
    iter_records_from_splits,
    records_to_corpus,
    require_versioned_corpus,
)
from llmg.data.temporalwiki import cl_query_from_row, load_tw_cl
from llmg.eval.temporal_metrics import subject_recall_at_k, temporal_recall_at_k
from llmg.memory.doc_catalog import DocCatalog
from llmg.search.harness import retrieve_bm25, retrieve_hybrid

if TYPE_CHECKING:
    from llmg.observability.run_session import RunSession


def _test_rows() -> tuple[list[dict[str, Any]], list[str], list[str], list[str], list[str]]:
    ds = load_tw_cl()
    rows = [ds["test"][i] for i in range(len(ds["test"]))]
    questions = [cl_query_from_row(r) for r in rows]
    gold_subjects = [r["subject_sitelink"] for r in rows]
    gold_answers = [r["object"] for r in rows]
    as_of = [str(r.get("snapshot_new") or "") for r in rows]
    return rows, questions, gold_subjects, gold_answers, as_of


def _train_corpus() -> tuple[dict[str, str], DocCatalog]:
    ds = load_tw_cl()
    records, _stats = dedupe_records(iter_records_from_splits(ds, ["train"]))
    return records_to_corpus(records), DocCatalog.from_records(records)


def _hard_indices(
    *,
    corpus: dict[str, str],
    catalog: DocCatalog,
    questions: list[str],
    gold_subjects: list[str],
    k: int,
) -> list[int]:
    doc_meta = catalog.doc_meta_tuple()
    hard: list[int] = []
    for i, (question, gold) in enumerate(zip(questions, gold_subjects, strict=True)):
        retrieved = retrieve_bm25(corpus, question, k)
        subjects = [doc_meta[d][0] for d in retrieved if d in doc_meta]
        if not subject_recall_at_k(subjects, gold, k):
            hard.append(i)
    return hard


def _eval_retrieval(
    *,
    search_mode: str,
    corpus: dict[str, str],
    catalog: DocCatalog,
    questions: list[str],
    gold_subjects: list[str],
    as_of: list[str],
    k: int,
) -> tuple[float, float]:
    if not questions:
        return 0.0, 0.0
    doc_meta = catalog.doc_meta_tuple()
    subject_hits = 0
    temporal_hits = 0
    for i, (question, gold) in enumerate(zip(questions, gold_subjects, strict=True)):
        retrieved = (
            retrieve_hybrid(corpus, question, k)
            if search_mode == "harness_hybrid"
            else retrieve_bm25(corpus, question, k)
        )
        subjects = [doc_meta[d][0] for d in retrieved if d in doc_meta]
        if subject_recall_at_k(subjects, gold, k):
            subject_hits += 1
        if temporal_recall_at_k(
            retrieved,
            gold_subject=gold,
            as_of=as_of[i] if i < len(as_of) else "",
            doc_meta=doc_meta,
            k=k,
        ):
            temporal_hits += 1
    n = len(questions)
    return subject_hits / n, temporal_hits / n


def _subset(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[i] for i in indices]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ensure_train_corpus_dir(session: RunSession) -> Path:
    root = session.run_dir / "corpus_train"
    if not corpus_is_current(root):
        export_all(corpus_root=root, index_splits=["train"], ds=load_tw_cl())
    require_versioned_corpus(root)
    return root


def run(
    *,
    k: int = 5,
    agent_model: str = "google/gemma-4-E4B-it",
    max_agent_steps: int = 16,
    max_hard_rows: int | None = 10,
    run_agent: bool | None = None,
    session: RunSession | None = None,
) -> dict[str, float]:
    if session is None:
        raise ValueError("P0-TW-05 requires RunSession")
    logger = session.logger
    run_phase = session.run_phase or "calibrate"
    should_run_agent = (run_phase == "official") if run_agent is None else bool(run_agent)

    corpus, catalog = _train_corpus()
    rows, questions, gold_subjects, gold_answers, as_of = _test_rows()
    hard_all = _hard_indices(
        corpus=corpus,
        catalog=catalog,
        questions=questions,
        gold_subjects=gold_subjects,
        k=k,
    )
    hard_eval = hard_all if max_hard_rows is None else hard_all[: max(0, int(max_hard_rows))]
    logger.info(
        "BM25-hard subset: %d/%d rows; evaluating %d",
        len(hard_all),
        len(rows),
        len(hard_eval),
    )

    hard_questions = _subset(questions, hard_eval)
    hard_gold_subjects = _subset(gold_subjects, hard_eval)
    hard_gold_answers = _subset(gold_answers, hard_eval)
    hard_as_of = _subset(as_of, hard_eval)
    hard_rows = [
        {
            "test_index": i,
            "question": questions[i],
            "gold_subject": gold_subjects[i],
            "gold_answer": gold_answers[i],
            "as_of": as_of[i],
        }
        for i in hard_eval
    ]
    (session.run_dir / "hard_subset.json").write_text(
        json.dumps(hard_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matrix_rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {
        "hard_subset_size": float(len(hard_all)),
        "eval_rows": float(len(hard_eval)),
    }
    for search_mode in ("harness_bm25", "harness_hybrid"):
        t0 = time.monotonic()
        recall, temporal = _eval_retrieval(
            search_mode=search_mode,
            corpus=corpus,
            catalog=catalog,
            questions=hard_questions,
            gold_subjects=hard_gold_subjects,
            as_of=hard_as_of,
            k=k,
        )
        elapsed = time.monotonic() - t0
        row = {
            "cell_id": f"memory_train_{search_mode}_test_bm25_hard",
            "search_mode": search_mode,
            "eval_rows": len(hard_eval),
            "retrieval_recall@k": recall,
            "temporal_recall@k": temporal,
            "elapsed_s": elapsed,
            "status": "ok",
        }
        matrix_rows.append(row)
        prefix = "hard_bm25" if search_mode == "harness_bm25" else "hard_hybrid"
        metrics[f"{prefix}_recall@{k}"] = float(recall)
        metrics[f"{prefix}_temporal@{k}"] = float(temporal)
        logger.info("%s recall@%d=%.4f temporal=%.4f", search_mode, k, recall, temporal)

    if should_run_agent:
        corpus_root = _ensure_train_corpus_dir(session)
        trace_dir = session.run_dir / "agent_traces" / "filesystem_train_agent_term_hybrid_test_bm25_hard"
        trace_dir.mkdir(parents=True, exist_ok=True)
        agent_metrics = run_agent_eval(
            corpus_root=corpus_root,
            questions=hard_questions,
            gold_subjects=hard_gold_subjects,
            gold_answers=hard_gold_answers,
            as_of_dates=hard_as_of,
            k=k,
            max_steps=max_agent_steps,
            max_rows=None,
            model_name=agent_model,
            trace_dir=trace_dir,
            agent_toolset="hybrid",
        )
        row = {
            "cell_id": "filesystem_train_agent_term_hybrid_test_bm25_hard",
            "search_mode": "agent_term_hybrid",
            "eval_rows": agent_metrics.get("eval_rows", 0.0),
            "retrieval_recall@k": agent_metrics.get(f"retrieval_recall@{k}", 0.0),
            "temporal_recall@k": agent_metrics.get(f"temporal_recall@{k}", 0.0),
            "answer_em": agent_metrics.get("answer_em", 0.0),
            "answer_cosine": agent_metrics.get("answer_cosine", 0.0),
            "answer_cosine_hit_rate": agent_metrics.get("answer_cosine_hit_rate", 0.0),
            "agent_steps_mean": agent_metrics.get("agent_steps_mean", 0.0),
            "status": "ok",
        }
        matrix_rows.append(row)
        metrics[f"hard_agent_hybrid_recall@{k}"] = float(row["retrieval_recall@k"])
        metrics[f"hard_agent_hybrid_temporal@{k}"] = float(row["temporal_recall@k"])
        metrics["answer_em"] = float(row["answer_em"])
        metrics["answer_cosine_hit_rate"] = float(row["answer_cosine_hit_rate"])
        metrics[f"retrieval_recall@{k}"] = float(row["retrieval_recall@k"])
    else:
        metrics[f"retrieval_recall@{k}"] = metrics[f"hard_hybrid_recall@{k}"]

    _write_tsv(session.run_dir / "matrix_results.tsv", matrix_rows)
    (session.run_dir / "matrix_results.json").write_text(
        json.dumps(matrix_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics
