from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .common import get_artifact_paths, resolve_path
from .inference import save_submission


def _get_member_paths(config: dict, member_run_name: str) -> tuple[str, str]:
    output_dir = resolve_path(config["paths"]["output_dir"])
    oof_path = output_dir / "oof" / f"{member_run_name}_oof.csv"
    test_path = output_dir / "oof" / f"{member_run_name}_test.csv"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing ensemble member artifacts for `{member_run_name}`.")
    return str(oof_path), str(test_path)


def _search_weights(step: float) -> list[float]:
    weights = np.arange(0.0, 1.0 + step, step)
    return [round(float(weight), 10) for weight in weights]


def _resolve_weight_candidates(raw_candidates: Sequence[float] | None, step: float) -> list[float]:
    if raw_candidates is None:
        return _search_weights(step)

    candidates = [round(float(weight), 10) for weight in raw_candidates]
    if not candidates:
        raise ValueError("`ensemble.weight_candidates` must contain at least one value.")
    if any(weight < 0.0 or weight > 1.0 for weight in candidates):
        raise ValueError("`ensemble.weight_candidates` values must be between 0.0 and 1.0.")
    return candidates


def _build_prediction_frame(
    member_frames: list[pd.DataFrame],
    members: list[str],
    id_col: str,
    target_col: str | None = None,
) -> pd.DataFrame:
    if not member_frames:
        raise ValueError("At least one ensemble member is required.")

    base_columns = [id_col]
    if target_col is not None:
        base_columns.append(target_col)

    merged = member_frames[0][base_columns + ["prediction"]].rename(
        columns={"prediction": f"{members[0]}_prediction"}
    )
    for member_name, frame in zip(members[1:], member_frames[1:]):
        merged = merged.merge(
            frame[[id_col, "prediction"]].rename(columns={"prediction": f"{member_name}_prediction"}),
            on=id_col,
            how="inner",
            validate="one_to_one",
        )
    return merged


def _resolve_manual_weights(raw_weights: dict | Sequence[float], members: list[str]) -> dict[str, float]:
    if isinstance(raw_weights, dict):
        weight_map = {member: float(raw_weights[member]) for member in members}
    elif isinstance(raw_weights, Sequence) and not isinstance(raw_weights, (str, bytes)):
        if len(raw_weights) != len(members):
            raise ValueError("`ensemble.weights` length must match the number of members.")
        weight_map = {member: float(weight) for member, weight in zip(members, raw_weights)}
    else:
        raise TypeError("`ensemble.weights` must be a dict or a list of floats.")

    total_weight = sum(weight_map.values())
    if total_weight <= 0:
        raise ValueError("`ensemble.weights` must sum to a positive value.")

    normalized = {member: weight / total_weight for member, weight in weight_map.items()}
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError("`ensemble.weights` must be non-negative.")
    return normalized


def _search_two_member_weights(
    base_oof: pd.DataFrame,
    members: list[str],
    target_col: str,
    candidates: list[float],
) -> tuple[dict[str, float], float]:
    best_weight = None
    best_mae = float("inf")
    for weight in candidates:
        blended = (
            weight * base_oof[f"{members[0]}_prediction"]
            + (1.0 - weight) * base_oof[f"{members[1]}_prediction"]
        )
        score = float(mean_absolute_error(base_oof[target_col], blended))
        if score < best_mae:
            best_mae = score
            best_weight = weight

    if best_weight is None:
        raise ValueError("Failed to find ensemble weights.")

    return (
        {
            members[0]: best_weight,
            members[1]: round(1.0 - best_weight, 10),
        },
        best_mae,
    )


def _blend_predictions(
    frame: pd.DataFrame,
    members: list[str],
    weights: dict[str, float],
    prediction_space: str = "raw",
    prediction_power: float | None = None,
    prediction_offset: float = 1.0,
    prediction_clip_min: float | None = None,
) -> pd.Series:
    blended = pd.Series(np.zeros(len(frame), dtype="float64"), index=frame.index)
    if prediction_space == "raw":
        for member in members:
            blended += weights[member] * frame[f"{member}_prediction"]
    elif prediction_space == "log1p":
        for member in members:
            member_prediction = frame[f"{member}_prediction"].astype("float64").clip(lower=0.0)
            blended += weights[member] * np.log1p(member_prediction)
        blended = np.expm1(blended)
    elif prediction_space == "power":
        if prediction_power is None:
            raise ValueError("`ensemble.prediction_power` is required when prediction_space is `power`.")
        if prediction_offset <= 0.0 and prediction_power <= 0.0:
            raise ValueError("`ensemble.prediction_offset` must be positive for non-positive powers.")
        for member in members:
            member_prediction = frame[f"{member}_prediction"].astype("float64").clip(lower=0.0)
            shifted_prediction = member_prediction + prediction_offset
            if prediction_power == 0.0:
                blended += weights[member] * np.log(shifted_prediction)
            else:
                blended += weights[member] * np.power(shifted_prediction, prediction_power)
        if prediction_power == 0.0:
            blended = np.exp(blended) - prediction_offset
        else:
            blended = np.power(np.maximum(blended, 0.0), 1.0 / prediction_power) - prediction_offset
    else:
        raise ValueError("`ensemble.prediction_space` must be `raw`, `log1p`, or `power`.")

    if prediction_clip_min is not None:
        blended = blended.clip(lower=prediction_clip_min)
    return blended


def run_ensemble(config: dict) -> str:
    id_col = config["columns"]["id_col"]
    target_col = config["columns"]["target_col"]
    members = config["ensemble"]["members"]
    step = float(config["ensemble"].get("weight_grid_step", 0.05))
    run_name = config["run_name"]
    manual_weights = config["ensemble"].get("weights")
    prediction_space = str(config["ensemble"].get("prediction_space", "raw"))
    prediction_power = config["ensemble"].get("prediction_power")
    if prediction_power is not None:
        prediction_power = float(prediction_power)
    prediction_offset = float(config["ensemble"].get("prediction_offset", 1.0))
    prediction_clip_min = config["ensemble"].get("prediction_clip_min")
    if prediction_clip_min is not None:
        prediction_clip_min = float(prediction_clip_min)
    weight_candidates = _resolve_weight_candidates(
        config["ensemble"].get("weight_candidates"),
        step,
    )

    if len(members) < 2:
        raise ValueError("Ensemble requires at least two members.")

    member_frames = []
    member_test_frames = []
    for member in members:
        oof_path, test_path = _get_member_paths(config, member)
        member_frames.append(pd.read_csv(oof_path, encoding="utf-8-sig"))
        member_test_frames.append(pd.read_csv(test_path, encoding="utf-8-sig"))

    base_oof = _build_prediction_frame(member_frames, members, id_col, target_col=target_col)
    base_test = _build_prediction_frame(member_test_frames, members, id_col, target_col=None)

    if manual_weights is not None:
        best_weights = _resolve_manual_weights(manual_weights, members)
        blended_oof = _blend_predictions(
            base_oof,
            members,
            best_weights,
            prediction_space=prediction_space,
            prediction_power=prediction_power,
            prediction_offset=prediction_offset,
            prediction_clip_min=prediction_clip_min,
        )
        best_mae = float(mean_absolute_error(base_oof[target_col], blended_oof))
    else:
        if prediction_space != "raw" or prediction_clip_min is not None:
            raise NotImplementedError(
                "Automatic weight search is only supported for raw prediction-space without clipping."
            )
        if len(members) != 2:
            raise NotImplementedError(
                "For three or more ensemble members, provide explicit `ensemble.weights`."
            )
        best_weights, best_mae = _search_two_member_weights(
            base_oof,
            members,
            target_col,
            weight_candidates,
        )
        blended_oof = _blend_predictions(base_oof, members, best_weights)

    blended_test = _blend_predictions(
        base_test,
        members,
        best_weights,
        prediction_space=prediction_space,
        prediction_power=prediction_power,
        prediction_offset=prediction_offset,
        prediction_clip_min=prediction_clip_min,
    )

    artifacts = get_artifact_paths(config)
    oof_output = base_oof[[id_col, target_col]].copy()
    oof_output["prediction"] = blended_oof.astype("float32")
    oof_output.to_csv(artifacts["oof_path"], index=False, encoding="utf-8-sig")

    test_output = base_test[[id_col]].copy()
    test_output["prediction"] = blended_test.astype("float32")
    test_output.to_csv(artifacts["test_pred_path"], index=False, encoding="utf-8-sig")

    metrics = {
        "run_name": run_name,
        "members": members,
        "weights": {member: round(weight, 10) for member, weight in best_weights.items()},
        "prediction_space": prediction_space,
        "prediction_power": prediction_power,
        "prediction_offset": prediction_offset,
        "prediction_clip_min": prediction_clip_min,
        "weight_candidates": weight_candidates if manual_weights is None else None,
        "oof_mae": best_mae,
    }
    with artifacts["metrics_path"].open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    weight_text = " ".join(f"{member}_weight={best_weights[member]:.2f}" for member in members)
    submission_path = save_submission(config)
    print(f"[ensemble] run={run_name} {weight_text} oof_mae={best_mae:.6f}")
    return submission_path
