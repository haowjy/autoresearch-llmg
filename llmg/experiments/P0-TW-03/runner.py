"""P0-TW-03 — phased baseline matrix (Wave A harness, Wave B agent, Wave C smoke)."""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llmg.agent.gemma_loop import run_agent_eval
from llmg.agent.tools import AgentToolset
from llmg.data.corpus_export import (
    corpus_is_current,
    dedupe_records,
    export_all,
    index_mode_to_splits,
    iter_records_from_splits,
    records_to_corpus,
    require_versioned_corpus,
)
from llmg.data.temporalwiki import load_tw_easy
from llmg.memory.doc_catalog import DocCatalog
from llmg.search.harness import run_harness_cell

if TYPE_CHECKING:
    from llmg.observability.run_session import RunSession

log = logging.getLogger("llmg.run")

AGENT_SEARCH_MODES = frozenset({"agent_term_basic", "agent_term_hybrid"})


def _agent_toolset(search_mode: str) -> AgentToolset:
    if search_mode == "agent_term_hybrid":
        return "hybrid"
    return "shell"


def _cell_id(cell: dict[str, Any]) -> str:
    parts = [
        cell.get("data_format", "?"),
        cell.get("index_mode", "?"),
        cell.get("search_mode", "?"),
        cell.get("eval_split", "?"),
    ]
    if cell.get("note"):
        parts.append(str(cell["note"]))
    return "_".join(parts)


def _eval_rows(
    ds,
    eval_split: str,
    *,
    max_rows: int | None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    split = ds[eval_split]
    n = len(split)
    if max_rows is not None:
        n = min(n, max_rows)
    questions = list(split["question"])[:n]
    gold_subjects = list(split["subject_sitelink"])[:n]
    gold_answers = list(split["object"])[:n]
    snapshots = list(split["snapshot_new"])[:n]
    as_of = [str(s or "") for s in snapshots]
    return questions, gold_subjects, gold_answers, as_of


def _corpus_dir(run_dir: Path, index_mode: str) -> Path:
    return run_dir / f"corpus_{index_mode}"


def _ensure_corpus(
    ds,
    run_dir: Path,
    index_mode: str,
    cache: dict[str, Path],
    *,
    force_reexport: bool = False,
) -> tuple[Path, dict[str, str], DocCatalog]:
    splits = index_mode_to_splits(index_mode)
    root = _corpus_dir(run_dir, index_mode)
    if force_reexport or not corpus_is_current(root):
        export_all(corpus_root=root, index_splits=splits, ds=ds)
    require_versioned_corpus(root)
    cache[index_mode] = root
    raw = iter_records_from_splits(ds, splits)
    records, _stats = dedupe_records(raw)
    catalog = DocCatalog.from_records(records)
    return root, records_to_corpus(records), catalog


def _run_harness_cell(
    cell: dict[str, Any],
    *,
    ds,
    run_dir: Path,
    k: int,
    max_rows: int | None,
    corpus_cache: dict[str, Path],
    force_reexport: bool = False,
) -> dict[str, Any]:
    index_mode = cell["index_mode"]
    corpus_root, memory_corpus, catalog = _ensure_corpus(
        ds, run_dir, index_mode, corpus_cache, force_reexport=force_reexport
    )
    questions, gold_subjects, _, as_of = _eval_rows(ds, cell["eval_split"], max_rows=max_rows)

    data_format = cell["data_format"]
    search_mode = cell["search_mode"]
    if search_mode == "harness_rg" and data_format != "filesystem":
        return {
            "cell_id": _cell_id(cell),
            "status": "skipped",
            "error": "harness_rg requires filesystem",
        }

    mem = memory_corpus if data_format == "memory" else None
    result = run_harness_cell(
        search_mode=search_mode,
        data_format=data_format,
        queries=questions,
        gold_subjects=gold_subjects,
        as_of_dates=as_of,
        k=k,
        catalog=catalog,
        memory_corpus=mem,
        corpus_root=corpus_root if data_format in ("filesystem", "sqlite") else None,
    )
    return {
        "cell_id": _cell_id(cell),
        "data_format": data_format,
        "index_mode": index_mode,
        "search_mode": search_mode,
        "eval_split": cell["eval_split"],
        "retrieval_recall@k": result.retrieval_recall_at_k,
        "temporal_recall@k": result.temporal_recall_at_k,
        "elapsed_s": result.elapsed_s,
        "eval_rows": len(questions),
        "status": "ok",
    }


def _run_agent_cell(
    cell: dict[str, Any],
    *,
    ds,
    run_dir: Path,
    k: int,
    max_agent_steps: int,
    agent_model: str,
    agent_heuristic_bootstrap: bool,
    max_rows: int | None,
    corpus_cache: dict[str, Path],
    force_reexport: bool = False,
) -> dict[str, Any]:
    index_mode = cell["index_mode"]
    corpus_root, _, _catalog = _ensure_corpus(
        ds, run_dir, index_mode, corpus_cache, force_reexport=force_reexport
    )
    questions, gold_subjects, gold_answers, as_of = _eval_rows(
        ds, cell["eval_split"], max_rows=max_rows
    )
    trace_dir = run_dir / "agent_traces" / _cell_id(cell)
    trace_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_agent_eval(
        corpus_root=corpus_root,
        questions=questions,
        gold_subjects=gold_subjects,
        gold_answers=gold_answers,
        as_of_dates=as_of,
        k=k,
        max_steps=max_agent_steps,
        max_rows=max_rows,
        model_name=agent_model,
        trace_dir=trace_dir,
        heuristic_bootstrap=agent_heuristic_bootstrap,
        agent_toolset=_agent_toolset(cell["search_mode"]),
    )
    return {
        "cell_id": _cell_id(cell),
        "data_format": cell["data_format"],
        "index_mode": index_mode,
        "search_mode": cell["search_mode"],
        "eval_split": cell["eval_split"],
        "retrieval_recall@k": metrics.get(f"retrieval_recall@{k}", 0.0),
        "temporal_recall@k": metrics.get(f"temporal_recall@{k}", 0.0),
        "answer_em": metrics.get("answer_em", 0.0),
        "answer_cosine": metrics.get("answer_cosine", 0.0),
        "answer_cosine_hit_rate": metrics.get("answer_cosine_hit_rate", 0.0),
        "agent_steps_mean": metrics.get("agent_steps_mean", 0.0),
        "cmd_count_mean": metrics.get("cmd_count_mean", 0.0),
        "bytes_read_mean": metrics.get("bytes_read_mean", 0.0),
        "eval_rows": metrics.get("eval_rows", 0.0),
        "elapsed_s": 0.0,
        "status": "ok",
    }


def _write_matrix_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(
    *,
    k: int = 5,
    max_agent_steps: int = 8,
    agent_model: str = "google/gemma-4-E4B-it",
    agent_heuristic_bootstrap: bool = False,
    max_eval_rows: int | None = None,
    waves: list[str] | None = None,
    skip_agent: bool = False,
    wave_a_cells: list[dict[str, Any]] | None = None,
    wave_b_cells: list[dict[str, Any]] | None = None,
    wave_c_max_eval_rows: int = 30,
    session: RunSession | None = None,
) -> dict[str, float]:
    logger = session.logger if session else log
    run_dir = session.run_dir if session else Path("llmg/runs/dev_P0-TW-03")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_phase = "calibrate"
    if session and session.run_phase:
        run_phase = session.run_phase

    ds = load_tw_easy()
    corpus_cache: dict[str, Path] = {}
    matrix_rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    force_reexport = run_phase == "official"

    active_waves = waves
    if active_waves is None:
        if run_phase == "official":
            active_waves = ["B"]
        else:
            active_waves = ["A", "C"]

    wave_a = wave_a_cells or []
    wave_b = wave_b_cells or []

    if "A" in active_waves:
        logger.info("Wave A: %d harness cells (full splits)", len(wave_a))
        for cell in wave_a:
            if cell.get("search_mode") in AGENT_SEARCH_MODES:
                continue
            cid = _cell_id(cell)
            logger.info("cell %s", cid)
            t_cell = time.monotonic()
            row = _run_harness_cell(
                cell,
                ds=ds,
                run_dir=run_dir,
                k=k,
                max_rows=None,
                corpus_cache=corpus_cache,
                force_reexport=force_reexport,
            )
            row["wave"] = "A"
            row["elapsed_s"] = time.monotonic() - t_cell
            matrix_rows.append(row)
            logger.info(
                "  recall@%d=%.4f elapsed=%.1fs",
                k,
                row.get("retrieval_recall@k", 0),
                row["elapsed_s"],
            )

    if "C" in active_waves and run_phase == "calibrate":
        logger.info("Wave C: smoke subset (max_eval_rows=%d)", wave_c_max_eval_rows)
        smoke_cells = wave_a[:2] if wave_a else []
        for cell in smoke_cells:
            if cell.get("search_mode") not in ("harness_bm25",):
                continue
            row = _run_harness_cell(
                cell,
                ds=ds,
                run_dir=run_dir,
                k=k,
                max_rows=wave_c_max_eval_rows,
                corpus_cache=corpus_cache,
                force_reexport=force_reexport,
            )
            row["wave"] = "C"
            row["cell_id"] = "smoke_" + row.get("cell_id", "")
            matrix_rows.append(row)

    if "B" in active_waves and not skip_agent:
        logger.info("Wave B: %d agent cells", len(wave_b))
        for cell in wave_b:
            cid = _cell_id(cell)
            logger.info("cell %s", cid)
            t_cell = time.monotonic()
            if cell.get("search_mode") in AGENT_SEARCH_MODES:
                row = _run_agent_cell(
                    cell,
                    ds=ds,
                    run_dir=run_dir,
                    k=k,
                    max_agent_steps=max_agent_steps,
                    agent_model=agent_model,
                    agent_heuristic_bootstrap=agent_heuristic_bootstrap,
                    max_rows=max_eval_rows,
                    corpus_cache=corpus_cache,
                    force_reexport=force_reexport,
                )
            else:
                row = _run_harness_cell(
                    cell,
                    ds=ds,
                    run_dir=run_dir,
                    k=k,
                    max_rows=max_eval_rows,
                    corpus_cache=corpus_cache,
                    force_reexport=force_reexport,
                )
            row["wave"] = "B"
            row["elapsed_s"] = time.monotonic() - t_cell
            matrix_rows.append(row)
            logger.info(
                "  recall@%d=%.4f temporal=%.4f em=%.4f cos=%.4f",
                k,
                row.get("retrieval_recall@k", 0),
                row.get("temporal_recall@k", 0),
                row.get("answer_em", 0),
                row.get("answer_cosine", 0),
            )

    matrix_path = run_dir / "matrix_results.tsv"
    _write_matrix_tsv(matrix_path, matrix_rows)
    (run_dir / "matrix_results.json").write_text(
        json.dumps(matrix_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("matrix written: %s (%d rows)", matrix_path, len(matrix_rows))

    metrics: dict[str, float] = {
        "matrix_cells": float(len(matrix_rows)),
        "elapsed_s": time.monotonic() - t0,
    }

    def _pin_key(prefix: str, row: dict[str, Any]) -> str:
        return f"{prefix}_{row['cell_id']}"

    for row in matrix_rows:
        if row.get("status") != "ok":
            continue
        rk = row.get("retrieval_recall@k")
        if rk is not None:
            metrics[_pin_key("recall", row)] = float(rk)
        if "answer_em" in row:
            metrics[_pin_key("answer_em", row)] = float(row["answer_em"])
        tr = row.get("temporal_recall@k")
        if tr is not None:
            metrics[_pin_key("temporal_recall", row)] = float(tr)
        ac = row.get("answer_cosine")
        if ac is not None:
            metrics[_pin_key("answer_cosine", row)] = float(ac)

    # Primary parity cells vs P0-TW-01 / 01b
    for row in matrix_rows:
        if row.get("status") != "ok":
            continue
        if (
            row.get("data_format") == "memory"
            and row.get("index_mode") == "train"
            and row.get("search_mode") == "harness_bm25"
            and row.get("eval_split") == "test"
            and row.get("wave") == "A"
        ):
            metrics["retrieval_recall@5"] = float(row["retrieval_recall@k"])
            metrics["temporal_recall@5"] = float(row.get("temporal_recall@k", 0))
        if (
            row.get("data_format") == "memory"
            and row.get("index_mode") == "train_stable"
            and row.get("search_mode") == "harness_bm25"
            and row.get("eval_split") == "stable"
            and row.get("wave") == "A"
        ):
            metrics["retrieval_recall@5_stable"] = float(row["retrieval_recall@k"])
            metrics["temporal_recall@5_stable"] = float(row.get("temporal_recall@k", 0))

    for row in matrix_rows:
        if row.get("status") != "ok" or row.get("wave") != "B":
            continue
        if row.get("search_mode") == "agent_term_basic" and row.get("eval_split") == "test":
            metrics["temporal_recall@5"] = float(row.get("temporal_recall@k", 0))
            metrics["answer_cosine"] = float(row.get("answer_cosine", 0))
            if "answer_cosine_hit_rate" in row:
                metrics["answer_cosine_hit_rate"] = float(row["answer_cosine_hit_rate"])
            break

    if "retrieval_recall@5" not in metrics and matrix_rows:
        ok = [r for r in matrix_rows if r.get("status") == "ok"]
        if ok:
            metrics["retrieval_recall@5"] = float(ok[0].get("retrieval_recall@k", 0))

    return metrics
