"""CLI entry: uv run python -m llmg.run --experiment <ID>"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from llmg.experiments.registry import (
    config_snapshot,
    list_experiment_ids,
    load_experiment,
    merge_params,
)
from llmg.observability import RunSession


def _parse_param(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected key=value, got {raw!r}")
    key, _, val = raw.partition("=")
    key = key.strip()
    val = val.strip()
    try:
        return key, json.loads(val)
    except json.JSONDecodeError:
        return key, val


def run_experiment(
    *,
    experiment_id: str,
    run_phase: str = "official",
    description: str = "",
    param_overrides: dict[str, Any] | None = None,
) -> int:
    spec = load_experiment(experiment_id)
    params = merge_params(spec, param_overrides or {})

    desc = description or spec.config.get("description", "")
    session = RunSession(
        experiment_id=experiment_id,
        run_phase=run_phase,
        description=desc,
    )
    session.log("spec_path=%s", spec.root)
    session.log("params=%s", params)

    snap = config_snapshot(spec, params)
    snap["run_phase"] = run_phase
    (session.run_dir / "experiment_config.json").write_text(
        json.dumps(snap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    primary = spec.primary_metric
    if not primary and "k" in params:
        primary = f"retrieval_recall@{params['k']}"

    try:
        metrics = spec.run_fn(session=session, **params)
        session.finish(metrics, primary_metric=primary)
    except Exception as exc:
        session.fail(exc)
        return 1
    return 0


def main() -> int:
    from llmg.util.hf_local import configure_hf_offline_if_requested

    configure_hf_offline_if_requested()
    ids = list_experiment_ids()
    parser = argparse.ArgumentParser(
        description="Run an LLMG experiment from llmg/experiments/<ID>/",
    )
    parser.add_argument(
        "--experiment",
        default=ids[0] if ids else None,
        help=f"Experiment ID (dirs under llmg/experiments/). Available: {', '.join(ids)}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiment IDs and exit",
    )
    parser.add_argument(
        "--run-phase",
        default="official",
        choices=("calibrate", "official"),
    )
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--param",
        action="append",
        dest="params",
        metavar="KEY=VALUE",
        help="Override config.yaml param (VALUE is JSON, e.g. k=10 or eval_split=\"stable\")",
    )
    args = parser.parse_args()

    if args.list:
        for eid in ids:
            spec = load_experiment(eid)
            print(f"{eid}\t{spec.run_mode}\t{spec.primary_metric}")
        return 0

    if not args.experiment:
        print("No experiments found under llmg/experiments/<ID>/", file=sys.stderr)
        return 1

    overrides: dict[str, Any] = {}
    if args.params:
        for raw in args.params:
            key, val = _parse_param(raw)
            overrides[key] = val

    return run_experiment(
        experiment_id=args.experiment,
        run_phase=args.run_phase,
        description=args.description,
        param_overrides=overrides,
    )


if __name__ == "__main__":
    raise SystemExit(main())
