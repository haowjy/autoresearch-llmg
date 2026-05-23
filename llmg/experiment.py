"""
Runner-editable entry for **calibrate** autoresearch loops.

Official gate runs should use locked `llmg/experiments/<ID>/config.yaml` via:

    uv run python -m llmg.run --experiment <ID>

During calibrate, the experiment-runner agent may edit THIS file to change
which experiment runs or to inline hyperparameter hacks, then:

    uv run python -m llmg.experiment > run.log 2>&1

Default: delegate to P0-TW-01 (change ACTIVE_EXPERIMENT when switching campaigns).
"""

from __future__ import annotations

ACTIVE_EXPERIMENT = "P0-TW-01"
RUN_PHASE = "calibrate"
DESCRIPTION = "calibrate via llmg.experiment.py"


def main() -> None:
    from llmg.run import run_experiment

    run_experiment(
        experiment_id=ACTIVE_EXPERIMENT,
        run_phase=RUN_PHASE,
        description=DESCRIPTION,
    )


if __name__ == "__main__":
    main()
