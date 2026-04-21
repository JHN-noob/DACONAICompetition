from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


LABEL_MAP = {
    "stable": 0,
    "unstable": 1,
    "0": 0,
    "1": 1,
    0: 0,
    1: 1,
}


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_records(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def parse_label(value: Any) -> int:
    key = value if value in LABEL_MAP else str(value).strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"Unsupported label value: {value}")
    return int(LABEL_MAP[key])


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config["paths"]["outputs"]
    return {
        "checkpoints": ensure_dir(outputs["checkpoints_dir"]),
        "oof": ensure_dir(outputs["oof_dir"]),
        "logs": ensure_dir(outputs["logs_dir"]),
        "calibration": ensure_dir(outputs["calibration_dir"]),
        "submissions": ensure_dir(outputs["submissions_dir"]),
    }


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.amp.GradScaler(device_type, enabled=enabled and device_type == "cuda")


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    best_score: float,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_history(path: str | Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def split_train_valid(records: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = records.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []

    for _, group in frame.groupby("label"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        valid_count = max(1, int(round(len(indices) * val_ratio)))
        valid_indices.extend(indices[:valid_count].tolist())
        train_indices.extend(indices[valid_count:].tolist())

    train_df = frame.loc[sorted(train_indices)].reset_index(drop=True)
    valid_df = frame.loc[sorted(valid_indices)].reset_index(drop=True)
    return train_df, valid_df


def build_stratified_folds(records: pd.DataFrame, n_splits: int, seed: int) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    frame = records.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    fold_assignments = np.empty(len(frame), dtype=np.int64)

    for _, group in frame.groupby("label"):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        chunks = np.array_split(indices, n_splits)
        for fold, chunk in enumerate(chunks):
            fold_assignments[chunk] = fold

    frame["fold"] = fold_assignments
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(n_splits):
        train_df = frame.loc[frame["fold"] != fold].drop(columns=["fold"]).reset_index(drop=True)
        valid_df = frame.loc[frame["fold"] == fold].drop(columns=["fold"]).reset_index(drop=True)
        folds.append((train_df, valid_df))
    return folds, frame[["id", "label", "fold"]].copy()

