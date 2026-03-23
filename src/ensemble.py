from __future__ import annotations

from datetime import datetime
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibrate import apply_calibration, fit_best_calibration
from .common import ensure_output_dirs, write_json
from .inference import make_submission
from .metrics import binary_logloss, logits_to_unstable_probs


def _generate_weight_candidates(
    num_models: int,
    step: float,
    min_weights: list[float] | None = None,
    max_weights: list[float] | None = None,
) -> list[tuple[float, ...]]:
    min_weights = min_weights or [0.0] * num_models
    max_weights = max_weights or [1.0] * num_models
    if num_models == 1:
        only = (1.0,)
        return [only] if min_weights[0] <= 1.0 <= max_weights[0] else []

    raw_values = [round(step * i, 10) for i in range(int(round(1.0 / step)) + 1)]
    candidates = []
    for values in itertools.product(raw_values, repeat=num_models):
        if abs(sum(values) - 1.0) < 1e-9 and all(
            lower <= value <= upper for value, lower, upper in zip(values, min_weights, max_weights)
        ):
            candidates.append(tuple(values))
    return candidates


def _load_model_entry(entry: dict[str, Any], split: str) -> pd.DataFrame:
    key = "oof_path" if split == "oof" else f"{split}_path"
    return pd.read_csv(entry[key])


def _load_calibration(entry: dict[str, Any]) -> dict[str, Any]:
    with Path(entry["calibration_path"]).open("r", encoding="utf-8") as file:
        return json.load(file)


def _get_weight_bounds(entries: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    min_weights = [float(entry.get("min_weight", 0.0)) for entry in entries]
    max_weights = [float(entry.get("max_weight", 1.0)) for entry in entries]
    return min_weights, max_weights


def _has_split_entries(entries: list[dict[str, Any]], split: str) -> bool:
    key = f"{split}_path"
    for entry in entries:
        if key not in entry:
            return False
        if not Path(entry[key]).exists():
            return False
    return True


def _build_aligned_frame(entries: list[dict[str, Any]], split: str) -> tuple[pd.DataFrame, list[str]]:
    base = _load_model_entry(entries[0], split=split).sort_values("id").reset_index(drop=True)
    labels = base["label"].to_numpy(dtype=np.float32) if "label" in base.columns else None
    merged = base[["id"]].copy()
    if labels is not None:
        merged["label"] = labels

    model_names = []
    for entry in entries:
        model_name = str(entry["name"])
        frame = _load_model_entry(entry, split=split).sort_values("id").reset_index(drop=True)
        calibration = _load_calibration(entry)
        merged[f"{model_name}_logit"] = apply_calibration(frame["raw_logit"].to_numpy(dtype=np.float32), calibration)
        model_names.append(model_name)
    return merged, model_names


def _resolve_submission_path(output_dir: Path, submission_cfg: dict[str, Any]) -> Path:
    requested_name = str(submission_cfg["name"])
    base_path = output_dir / requested_name
    if not bool(submission_cfg.get("append_timestamp", False)):
        return base_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = base_path.stem
    suffix = base_path.suffix
    candidate = output_dir / f"{stem}_{timestamp}{suffix}"
    duplicate_index = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}_{timestamp}_{duplicate_index}{suffix}"
        duplicate_index += 1
    return candidate


def _resolve_selection_split(config: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    requested = str(config["ensemble"].get("selection_split", "dev")).lower()
    if requested not in {"oof", "dev"}:
        raise ValueError(f"Unsupported ensemble selection_split: {requested}")
    if requested == "dev" and not _has_split_entries(entries, split="dev"):
        raise ValueError("Ensemble selection_split='dev' requires dev_path for every model entry.")
    return requested


def run_ensemble_oof(config: dict[str, Any]) -> dict[str, Any]:
    output_dirs = ensure_output_dirs(config)
    ensemble_name = str(config["experiment"]["name"])
    entries = list(config["models"])
    selection_split = _resolve_selection_split(config, entries)
    aligned_frame, model_names = _build_aligned_frame(entries, split=selection_split)
    min_weights, max_weights = _get_weight_bounds(entries)

    labels = aligned_frame["label"].to_numpy(dtype=np.float32)
    logits_matrix = np.stack([aligned_frame[f"{name}_logit"].to_numpy(dtype=np.float32) for name in model_names], axis=1)

    best_weights = None
    best_logloss = float("inf")
    for weights in _generate_weight_candidates(
        len(model_names),
        float(config["ensemble"]["weight_step"]),
        min_weights=min_weights,
        max_weights=max_weights,
    ):
        ensemble_logits = logits_matrix @ np.asarray(weights, dtype=np.float32)
        score = binary_logloss(labels, logits_to_unstable_probs(ensemble_logits))
        if score < best_logloss:
            best_logloss = score
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("Failed to generate ensemble weights.")

    ensemble_logits = logits_matrix @ np.asarray(best_weights, dtype=np.float32)
    summary = {
        "model_names": model_names,
        "weights": list(best_weights),
        "min_weights": min_weights,
        "max_weights": max_weights,
        "selection_split": selection_split,
        "selection_logloss_before_final_calibration": best_logloss,
    }
    if selection_split == "oof":
        summary["oof_logloss_before_final_calibration"] = best_logloss

    final_calibration = None
    if bool(config["ensemble"].get("final_calibration", True)):
        final_calibration = fit_best_calibration(
            logits=ensemble_logits,
            labels=labels,
            calibration_cfg=config["calibration"],
        )
        ensemble_logits = apply_calibration(ensemble_logits, final_calibration)
        summary["final_calibration"] = final_calibration
        summary["selection_logloss_after_final_calibration"] = binary_logloss(
            labels,
            logits_to_unstable_probs(ensemble_logits),
        )
        if selection_split == "oof":
            summary["oof_logloss_after_final_calibration"] = summary["selection_logloss_after_final_calibration"]

    oof_frame = aligned_frame[["id", "label"]].copy()
    oof_frame["ensemble_logit"] = ensemble_logits
    oof_frame["ensemble_prob"] = logits_to_unstable_probs(ensemble_logits)

    oof_path = output_dirs["oof"] / f"{ensemble_name}_{selection_split}.csv"
    summary_path = output_dirs["logs"] / f"{ensemble_name}_summary.json"
    oof_frame.to_csv(oof_path, index=False)
    write_json(summary_path, summary)
    return {
        "ensemble_selection_path": str(oof_path),
        "summary_path": str(summary_path),
    }


def run_ensemble_submission(config: dict[str, Any]) -> dict[str, Any]:
    output_dirs = ensure_output_dirs(config)
    ensemble_name = str(config["experiment"]["name"])
    summary_path = output_dirs["logs"] / f"{ensemble_name}_summary.json"
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    entries = list(config["models"])
    weights = np.asarray(summary["weights"], dtype=np.float32)
    final_calibration = summary.get("final_calibration")

    result: dict[str, Any] = {}

    if _has_split_entries(entries, split="dev"):
        dev_frame, model_names = _build_aligned_frame(entries, split="dev")
        dev_logits = np.stack([dev_frame[f"{name}_logit"].to_numpy(dtype=np.float32) for name in model_names], axis=1) @ weights
        if final_calibration is not None:
            dev_logits = apply_calibration(dev_logits, final_calibration)
        dev_frame = dev_frame[["id", "label"]].copy()
        dev_frame["ensemble_logit"] = dev_logits
        dev_frame["ensemble_prob"] = logits_to_unstable_probs(dev_logits)
        dev_logloss = binary_logloss(
            dev_frame["label"].to_numpy(dtype=np.float32),
            dev_frame["ensemble_prob"].to_numpy(dtype=np.float32),
        )
        dev_path = output_dirs["submissions"] / f"{ensemble_name}_dev_predictions.csv"
        dev_frame.to_csv(dev_path, index=False)
        print(f"[ensemble] dev_logloss={dev_logloss:.6f} saved_dev_predictions={dev_path}")
        result["dev_prediction_path"] = str(dev_path)
        result["dev_logloss"] = dev_logloss
    else:
        print("[ensemble] external dev predictions not found, skip dev logloss export")

    test_frame, model_names = _build_aligned_frame(entries, split="test")
    test_logits = np.stack([test_frame[f"{name}_logit"].to_numpy(dtype=np.float32) for name in model_names], axis=1) @ weights
    if final_calibration is not None:
        test_logits = apply_calibration(test_logits, final_calibration)
    prediction_frame = test_frame[["id"]].copy()
    prediction_frame["prob"] = logits_to_unstable_probs(test_logits)
    submission_path = _resolve_submission_path(output_dirs["submissions"], config["submission"])
    submission_clip_eps = float(config["submission"].get("clip_eps", 1e-4))
    make_submission(prediction_frame, submission_path, clip_eps=submission_clip_eps)
    print(f"[ensemble] submission_clip_eps={submission_clip_eps:.1e} saved_submission={submission_path}")

    result["submission_path"] = str(submission_path)
    return result
