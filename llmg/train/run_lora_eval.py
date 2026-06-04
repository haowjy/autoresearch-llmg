"""Subprocess entry: hybrid-agent eval with a saved LoRA adapter (frees GPU from training)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from llmg.agent.gemma_loop import run_agent_eval
from llmg.data.corpus_export import export_all, index_mode_to_splits
from llmg.data.temporalwiki import load_tw_easy
from llmg.util.timing import PhaseTimer, record_partial_phase

log = logging.getLogger(__name__)


def eval_rows_p1(
    ds,
    eval_split: str,
    *,
    max_rows: int | None,
    start_row: int = 0,
    relation_hint: bool = False,
) -> tuple[list[str], list[str], list[str], list[str], list[int]]:
    split = ds[eval_split]
    start = max(0, int(start_row))
    n = max(0, len(split) - start)
    if max_rows is not None:
        n = min(n, max_rows)
    rows = [split[start + i] for i in range(n)]
    questions = [_question_for_eval(r, relation_hint=relation_hint) for r in rows]
    gold_subjects = [str(r.get("subject_sitelink") or "") for r in rows]
    gold_answers = [str(r.get("object") or "") for r in rows]
    as_of = [str(r.get("snapshot_new") or "") for r in rows]
    row_indices = [start + i for i in range(n)]
    return questions, gold_subjects, gold_answers, as_of, row_indices


def _question_for_eval(row, *, relation_hint: object) -> str:
    question = str(row.get("question") or "")
    relation = str(row.get("relation") or "").strip()
    subject = str(row.get("subject_sitelink") or row.get("subject") or "").strip()
    mode = _relation_hint_mode(relation_hint)
    if not relation or mode == "off":
        return question
    if mode == "when_nondate" and not _needs_when_nondate_hint(question, relation):
        return question
    subject_text = f" for {subject}" if subject else ""
    hint = (
        f"Target relation: {relation}.\n"
        f"Return the full {relation} name/value{subject_text}. "
        "If that value is named in the question, submit the complete phrase from the question. "
        "Return only that short value, not a supporting date/year or explanation."
    )
    return f"{question}\n\n{hint}"


def _relation_hint_mode(raw: object) -> str:
    if raw is True:
        return "all"
    if raw in (False, None, ""):
        return "off"
    text = str(raw).strip().lower()
    if text in {"0", "false", "off", "none", "no"}:
        return "off"
    if text in {"1", "true", "all", "yes"}:
        return "all"
    if text in {"when", "when_nondate"}:
        return "when_nondate"
    raise ValueError(f"unsupported eval_relation_hint={raw!r}")


def _needs_when_nondate_hint(question: str, relation: str) -> bool:
    if not question.strip().lower().startswith("when"):
        return False
    lower_relation = relation.lower()
    date_terms = ("date", "time", "year", "inception", "dissolved", "start", "end")
    return not any(term in lower_relation for term in date_terms)


def eval_lora_run_dir(run_dir: Path) -> dict[str, float]:
    cfg_path = run_dir / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    params = cfg.get("params") or {}

    k = int(params.get("k", 5))
    eval_split = str(params.get("eval_split", "test"))
    base_model = str(params.get("base_model", "google/gemma-4-E4B-it"))
    agent_toolset = str(params.get("agent_toolset", "hybrid"))
    if agent_toolset not in {"hybrid", "hybrid_deep"}:
        raise ValueError(f"P1-02 LoRA eval supports hybrid toolsets only, got {agent_toolset!r}")
    corpus_index_mode = str(params.get("corpus_index_mode", "train_stable"))
    max_agent_steps = int(params.get("max_agent_steps", 16))
    search_snippets_per_hit_raw = params.get("search_snippets_per_hit")
    search_snippets_per_hit = (
        int(search_snippets_per_hit_raw)
        if search_snippets_per_hit_raw is not None
        else None
    )
    search_snippet_hits_raw = params.get("search_snippet_hits")
    search_snippet_hits = (
        int(search_snippet_hits_raw)
        if search_snippet_hits_raw is not None
        else None
    )
    eval_relation_hint = params.get("eval_relation_hint", False)
    eval_start_row = int(params.get("eval_start_row", 0))
    max_eval_rows = params.get("max_eval_rows")
    if max_eval_rows is None:
        phase = cfg.get("run_phase", "official")
        if phase == "calibrate":
            max_eval_rows = int(params.get("calibrate_max_eval_rows", 10))
    else:
        max_eval_rows = int(max_eval_rows)

    adapter_dir = adapter_dir_from_params(run_dir, params)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"missing adapter: {adapter_dir}")

    ds = load_tw_easy()
    splits = index_mode_to_splits(corpus_index_mode)
    corpus_root = run_dir / f"corpus_{corpus_index_mode}"
    if not corpus_root.is_dir():
        export_all(corpus_root=corpus_root, index_splits=splits, ds=ds)

    questions, gold_subjects, gold_answers, as_of, row_indices = eval_rows_p1(
        ds,
        eval_split,
        max_rows=max_eval_rows,
        start_row=eval_start_row,
        relation_hint=eval_relation_hint,
    )
    trace_dir = (
        run_dir
        / "agent_traces"
        / f"filesystem_{corpus_index_mode}_agent_term_{agent_toolset}_{eval_split}_lora"
    )
    trace_dir.mkdir(parents=True, exist_ok=True)

    timer = PhaseTimer(record_gpu=True)
    timer.start("lora_eval")
    try:
        metrics = run_agent_eval(
            corpus_root=corpus_root,
            questions=questions,
            gold_subjects=gold_subjects,
            gold_answers=gold_answers,
            as_of_dates=as_of,
            k=k,
            max_steps=max_agent_steps,
            max_rows=max_eval_rows,
            model_name=base_model,
            adapter_path=adapter_dir,
            trace_dir=trace_dir,
            agent_toolset=agent_toolset,  # type: ignore[arg-type]
            search_snippets_per_hit=search_snippets_per_hit,
            search_snippet_hits=search_snippet_hits,
            row_indices=row_indices,
        )
    finally:
        seconds = timer.stop("lora_eval")
        record_partial_phase(run_dir, "lora_eval", seconds, record_gpu=True)
    metrics["eval_rows"] = float(len(questions))
    return metrics


def adapter_dir_from_params(run_dir: Path, params: dict) -> Path:
    raw = params.get("adapter_source_run")
    if not raw:
        return run_dir / "lora_adapter"
    source = Path(str(raw))
    if not source.is_absolute():
        source = Path("llmg/runs") / source
    if source.name != "lora_adapter":
        source = source / "lora_adapter"
    return source


def main() -> None:
    from llmg.util.hf_local import configure_hf_offline_if_requested

    configure_hf_offline_if_requested()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    metrics = eval_lora_run_dir(args.run_dir)
    out = args.run_dir / "lora_eval_metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
