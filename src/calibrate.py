from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .common import ensure_output_dirs, write_json
from .metrics import binary_logloss, logits_to_unstable_probs


def apply_calibration(logits: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    method = calibration["best_method"]
    if method == "none":
        return logits
    if method == "bias":
        return logits + float(calibration["b"])
    if method == "temperature":
        return logits / float(calibration["temperature"])
    if method in {"affine", "platt"}:
        return logits * float(calibration["a"]) + float(calibration["b"])
    raise ValueError(f"Unsupported calibration method: {method}")


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray, candidates: list[float]) -> dict[str, Any]:
    best_temperature = 1.0
    best_logloss = float("inf")
    for temperature in candidates:
        calibrated_logits = logits / float(temperature)
        score = binary_logloss(labels, logits_to_unstable_probs(calibrated_logits))
        if score < best_logloss:
            best_logloss = score
            best_temperature = float(temperature)
    return {
        "method": "temperature",
        "temperature": best_temperature,
        "logloss": best_logloss,
    }


def fit_affine_calibration(logits: np.ndarray, labels: np.ndarray, max_iter: int) -> dict[str, Any]:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(a * logits_t + b, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    calibrated_logits = (a.detach() * logits_t + b.detach()).cpu().numpy()
    score = binary_logloss(labels, logits_to_unstable_probs(calibrated_logits))
    return {
        "method": "affine",
        "a": float(a.detach().cpu().item()),
        "b": float(b.detach().cpu().item()),
        "logloss": score,
    }


def fit_bias_calibration(logits: np.ndarray, labels: np.ndarray, max_iter: int) -> dict[str, Any]:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([b], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits_t + b, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    calibrated_logits = (logits_t + b.detach()).cpu().numpy()
    score = binary_logloss(labels, logits_to_unstable_probs(calibrated_logits))
    return {
        "method": "bias",
        "b": float(b.detach().cpu().item()),
        "logloss": score,
    }


def fit_best_calibration(logits: np.ndarray, labels: np.ndarray, calibration_cfg: dict[str, Any]) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    before_logloss = binary_logloss(labels, logits_to_unstable_probs(logits))
    allowed_methods = [str(method).lower() for method in calibration_cfg.get("allowed_methods", ["affine", "temperature"])]
    results: list[dict[str, Any]] = []

    if "none" in allowed_methods:
        results.append(
            {
                "method": "none",
                "logloss": before_logloss,
            }
        )
    if "temperature" in allowed_methods:
        results.append(
            fit_temperature_scaling(
                logits=logits,
                labels=labels,
                candidates=[float(value) for value in calibration_cfg["temperature_candidates"]],
            )
        )
    if "bias" in allowed_methods:
        results.append(
            fit_bias_calibration(
                logits=logits,
                labels=labels,
                max_iter=int(calibration_cfg.get("bias_max_iter", 200)),
            )
        )
    if "affine" in allowed_methods or "platt" in allowed_methods:
        results.append(
            fit_affine_calibration(
                logits=logits,
                labels=labels,
                max_iter=int(calibration_cfg.get("affine_max_iter", calibration_cfg.get("platt_max_iter", 200))),
            )
        )
    if not results:
        raise ValueError("No supported calibration method selected.")

    best_result = min(results, key=lambda result: float(result["logloss"]))
    calibration = {
        "before_logloss": before_logloss,
        "after_logloss": best_result["logloss"],
        "best_method": best_result["method"],
    }
    if best_result["method"] == "none":
        return calibration
    if best_result["method"] == "bias":
        calibration["b"] = best_result["b"]
        return calibration
    if best_result["method"] == "temperature":
        calibration["temperature"] = best_result["temperature"]
    else:
        calibration["a"] = best_result["a"]
        calibration["b"] = best_result["b"]
    return calibration


def _load_selection_frame(
    config: dict[str, Any],
    output_dirs: dict[str, Path],
    experiment_name: str,
    oof_path: Path,
) -> tuple[pd.DataFrame, Path, str]:
    selection_split = str(config["calibration"].get("selection_split", "oof")).lower()
    if selection_split == "dev":
        selection_path = output_dirs["oof"] / f"{experiment_name}_dev_mean.csv"
        return pd.read_csv(selection_path), selection_path, selection_split
    return pd.read_csv(oof_path), oof_path, "oof"


def run_calibration(config: dict[str, Any], oof_csv_path: str | Path | None = None) -> dict[str, Any]:
    output_dirs = ensure_output_dirs(config)
    experiment_name = str(config["experiment"]["name"])
    oof_path = Path(oof_csv_path) if oof_csv_path is not None else output_dirs["oof"] / f"{experiment_name}_oof.csv"
    oof_frame = pd.read_csv(oof_path)
    selection_frame, selection_path, selection_split = _load_selection_frame(config, output_dirs, experiment_name, oof_path)
    calibration = fit_best_calibration(
        logits=selection_frame["raw_logit"].to_numpy(dtype=np.float32),
        labels=selection_frame["label"].to_numpy(dtype=np.float32),
        calibration_cfg=config["calibration"],
    )
    calibration["selection_split"] = selection_split
    calibration["selection_path"] = str(selection_path)

    calibration_path = output_dirs["calibration"] / f"{experiment_name}_calibration.json"
    write_json(calibration_path, calibration)

    calibrated_logits = apply_calibration(oof_frame["raw_logit"].to_numpy(dtype=np.float32), calibration)
    calibrated_frame = oof_frame.copy()
    calibrated_frame["calibrated_logit"] = calibrated_logits
    calibrated_frame["calibrated_prob"] = logits_to_unstable_probs(calibrated_logits)
    calibrated_oof_path = output_dirs["calibration"] / f"{experiment_name}_oof_calibrated.csv"
    calibrated_frame.to_csv(calibrated_oof_path, index=False)

    calibrated_dev_path = None
    dev_path = output_dirs["oof"] / f"{experiment_name}_dev_mean.csv"
    save_dev_predictions = bool(config["calibration"].get("save_dev_predictions", False))
    if save_dev_predictions and dev_path.exists():
        dev_frame = pd.read_csv(dev_path)
        dev_logits = apply_calibration(dev_frame["raw_logit"].to_numpy(dtype=np.float32), calibration)
        calibrated_dev = dev_frame.copy()
        calibrated_dev["calibrated_logit"] = dev_logits
        calibrated_dev["calibrated_prob"] = logits_to_unstable_probs(dev_logits)
        calibrated_dev_path = output_dirs["calibration"] / f"{experiment_name}_dev_calibrated.csv"
        calibrated_dev.to_csv(calibrated_dev_path, index=False)

    print(
        f"[calibration] model={experiment_name} split={selection_split} method={calibration['best_method']} "
        f"before={calibration['before_logloss']:.5f} after={calibration['after_logloss']:.5f}"
    )
    result = {
        "calibration_path": str(calibration_path),
        "calibrated_oof_path": str(calibrated_oof_path),
        "before_logloss": calibration["before_logloss"],
        "after_logloss": calibration["after_logloss"],
    }
    if calibrated_dev_path is not None:
        result["calibrated_dev_path"] = str(calibrated_dev_path)
    return result
