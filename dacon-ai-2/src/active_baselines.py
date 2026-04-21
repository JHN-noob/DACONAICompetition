from __future__ import annotations

import json

from .common import get_artifact_paths, load_config
from .ensemble import run_ensemble
from .run_pipeline import run_cv, run_submission
from .stacking import run_stacking, run_stacking_cv

ACTIVE_SINGLE_CONFIG = "configs/lightgbm_v021_no_layout_id_core_holdout_log1p_scenario_relative.yaml"
ACTIVE_BEST_SUBMISSION_CONFIG = "configs/blend_v053_v047_stack_v006_diversity.yaml"
ACTIVE_NEXT_CANDIDATE_CONFIG: str | None = None


def get_active_baseline_paths() -> dict[str, str | None]:
    return {
        "single": ACTIVE_SINGLE_CONFIG,
        "best_submission": ACTIVE_BEST_SUBMISSION_CONFIG,
        "next_candidate": ACTIVE_NEXT_CANDIDATE_CONFIG,
    }


def load_active_single_config() -> dict:
    return load_config(ACTIVE_SINGLE_CONFIG)


def load_active_best_submission_config() -> dict:
    return load_config(ACTIVE_BEST_SUBMISSION_CONFIG)


def load_active_next_candidate_config() -> dict:
    if ACTIVE_NEXT_CANDIDATE_CONFIG is None:
        raise ValueError("No active next candidate. The final submission model is v053.")
    return load_config(ACTIVE_NEXT_CANDIDATE_CONFIG)


def run_active_single_cv() -> dict:
    return run_cv(load_active_single_config())


def run_active_single_submission() -> str:
    return run_submission(load_active_single_config())


def run_active_best_submission() -> str:
    config = load_active_best_submission_config()
    if "stacking" in config:
        return run_stacking(config)
    if "ensemble" in config:
        return run_ensemble(config)
    return run_submission(config)


def run_active_next_candidate() -> str:
    raise ValueError("All experiments are closed. Use run_active_best_submission().")


def run_active_next_candidate_cv() -> dict:
    raise ValueError("All experiments are closed. Use run_active_best_submission().")


def run_config(config: dict) -> str:
    if "stacking" in config:
        return run_stacking(config)
    if "ensemble" in config:
        return run_ensemble(config)
    run_cv(config)
    return run_submission(config)


def run_config_cv(config: dict) -> dict:
    if "stacking" in config:
        return run_stacking_cv(config)
    if "ensemble" in config:
        run_ensemble(config)
        with get_artifact_paths(config)["metrics_path"].open("r", encoding="utf-8") as file:
            return json.load(file)
    return run_cv(config)
