from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def to_project_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    config = deepcopy(config)
    config.setdefault("seed", 42)
    config.setdefault("paths", {})
    config["paths"].setdefault("data_dir", "data")
    config["paths"].setdefault("output_dir", "outputs")
    config["_config_path"] = to_project_relative(config_path)
    return config


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def prepare_runtime(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    ensure_output_dirs(config)


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    output_dir = resolve_path(config["paths"]["output_dir"])
    paths = {
        "output_dir": output_dir,
        "oof_dir": output_dir / "oof",
        "model_dir": output_dir / "models",
        "log_dir": output_dir / "logs",
        "importance_dir": output_dir / "importance",
        "submission_dir": output_dir / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    run_name = config["run_name"]
    paths = ensure_output_dirs(config)
    paths.update(
        {
            "oof_path": paths["oof_dir"] / f"{run_name}_oof.csv",
            "test_pred_path": paths["oof_dir"] / f"{run_name}_test.csv",
            "metrics_path": paths["log_dir"] / f"{run_name}_metrics.json",
            "importance_path": paths["importance_dir"] / f"{run_name}_feature_importance.csv",
            "submission_path": paths["submission_dir"] / f"{run_name}_submission.csv",
        }
    )
    return paths
