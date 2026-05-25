"""P0-TW-04 — P0-TW-03 matrix on TemporalWiki drift CL (base / harder triples)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmg.baseline.tw_matrix_run import eval_rows_cl, run_matrix
from llmg.data.temporalwiki import load_tw_cl

if TYPE_CHECKING:
    from llmg.observability.run_session import RunSession


def run(
    *,
    k: int = 5,
    max_agent_steps: int = 16,
    agent_model: str = "google/gemma-4-E4B-it",
    max_eval_rows: int | None = None,
    waves: list[str] | None = None,
    official_waves: list[str] | None = None,
    skip_agent: bool = False,
    wave_a_cells: list[dict[str, Any]] | None = None,
    wave_b_cells: list[dict[str, Any]] | None = None,
    wave_c_max_eval_rows: int = 30,
    session: RunSession | None = None,
) -> dict[str, float]:
    return run_matrix(
        load_dataset_fn=load_tw_cl,
        eval_rows_fn=eval_rows_cl,
        dev_run_dir_name="dev_P0-TW-04",
        k=k,
        max_agent_steps=max_agent_steps,
        agent_model=agent_model,
        max_eval_rows=max_eval_rows,
        waves=waves,
        official_waves=official_waves,
        skip_agent=skip_agent,
        wave_a_cells=wave_a_cells,
        wave_b_cells=wave_b_cells,
        wave_c_max_eval_rows=wave_c_max_eval_rows,
        session=session,
    )
