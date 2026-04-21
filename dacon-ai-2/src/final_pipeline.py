from __future__ import annotations

from .common import load_config
from .ensemble import run_ensemble
from .run_pipeline import run_cv
from .stacking import run_stacking

FINAL_RUN_ORDER = [
    "configs/lightgbm_v013_no_layout_id_core_holdout_log1p.yaml",
    "configs/lightgbm_v014_no_layout_id_core_holdout_tuned_log1p.yaml",
    "configs/lightgbm_v015_no_layout_id_core_holdout_log1p_adv_weighted.yaml",
    "configs/stack_lgb_meta_v003_log1p_family.yaml",
    "configs/lightgbm_v021_no_layout_id_core_holdout_log1p_scenario_relative.yaml",
    "configs/stack_lgb_meta_v005_context_aware.yaml",
    "configs/lightgbm_v022_residual_stack_v005_context_gate.yaml",
    "configs/lightgbm_v029_residual_v021_context_gate.yaml",
    "configs/blend_v047_residual_v029_shrinkage.yaml",
    "configs/stack_lgb_meta_v006_strong_family_context_aware.yaml",
    "configs/blend_v053_v047_stack_v006_diversity.yaml",
]

FINAL_SUBMISSION_CONFIG = "configs/blend_v053_v047_stack_v006_diversity.yaml"


def _run_config(config: dict) -> str | dict:
    if "stacking" in config:
        return run_stacking(config)
    if "ensemble" in config:
        return run_ensemble(config)
    return run_cv(config)


def run_final_pipeline() -> str:
    final_submission_path = ""
    for config_path in FINAL_RUN_ORDER:
        config = load_config(config_path)
        result = _run_config(config)
        if config_path == FINAL_SUBMISSION_CONFIG:
            final_submission_path = str(result)

    if not final_submission_path:
        raise RuntimeError("Final submission was not generated.")
    return final_submission_path


def run_final_submission_from_existing_predictions() -> str:
    config = load_config(FINAL_SUBMISSION_CONFIG)
    return run_ensemble(config)
