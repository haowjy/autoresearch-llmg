"""Discover experiments under llmg/experiments/<ID>/ — no edits to llmg/run.py per experiment."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

EXPERIMENTS_ROOT = Path(__file__).resolve().parent
RunFn = Callable[..., dict[str, float]]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    root: Path
    config: dict[str, Any]
    run_fn: RunFn

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.config.get("params") or {})

    @property
    def primary_metric(self) -> str | None:
        return self.config.get("primary_metric")

    @property
    def run_mode(self) -> str:
        return self.config.get("run_mode", "eval_only")


def list_experiment_ids() -> list[str]:
    ids = []
    for path in sorted(EXPERIMENTS_ROOT.iterdir()):
        if path.is_dir() and _is_experiment_dir(path):
            ids.append(path.name)
    return ids


def load_experiment(experiment_id: str) -> ExperimentSpec:
    root = EXPERIMENTS_ROOT / experiment_id
    if not _is_experiment_dir(root):
        available = ", ".join(list_experiment_ids()) or "(none)"
        raise KeyError(
            f"Unknown experiment {experiment_id!r}. "
            f"Expected llmg/experiments/{experiment_id}/ with config.yaml + runner.py. "
            f"Available: {available}"
        )

    config_path = root / "config.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    config_id = config.get("experiment_id", experiment_id)
    if config_id != experiment_id:
        raise ValueError(
            f"config.yaml experiment_id={config_id!r} != directory {experiment_id!r}"
        )

    run_fn = _load_runner(root / "runner.py")
    return ExperimentSpec(
        experiment_id=experiment_id,
        root=root,
        config=config,
        run_fn=run_fn,
    )


def merge_params(spec: ExperimentSpec, overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {**spec.params}
    for key, val in overrides.items():
        if val is not None:
            merged[key] = val
    return merged


def _is_experiment_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.yaml").is_file() and (path / "runner.py").is_file()


def _load_runner(runner_path: Path) -> RunFn:
    spec = importlib.util.spec_from_file_location(
        f"llmg.experiments.{runner_path.parent.name}.runner",
        runner_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load runner: {runner_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{runner_path} must define run(session=None, **params)")
    return mod.run  # type: ignore[no-any-return]


def config_snapshot(spec: ExperimentSpec, params: dict[str, Any]) -> dict[str, Any]:
    """Full config written into each run dir."""
    return {
        "experiment_id": spec.experiment_id,
        "run_mode": spec.run_mode,
        "primary_metric": spec.primary_metric,
        "params": params,
        "spec_path": str(spec.root),
        "config_yaml": json.loads(json.dumps(spec.config)),  # YAML -> JSON-safe
    }
