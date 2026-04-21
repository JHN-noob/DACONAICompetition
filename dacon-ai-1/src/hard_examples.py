from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _row_logloss(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=np.float32).reshape(-1), eps, 1.0 - eps)
    return -(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))


def build_hard_example_frame(
    reference_oof_path: str | Path,
    top_percent: float | None = None,
    class_top_percents: dict[int, float] | None = None,
    class_duplicate_factors: dict[int, int] | None = None,
    default_duplicate_factor: int = 1,
) -> pd.DataFrame:
    frame = pd.read_csv(reference_oof_path)
    required = {"id", "label", "prob"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in hard-example source: {sorted(missing)}")

    output = frame[["id", "label", "prob"]].copy()
    output["label"] = output["label"].astype("int64")
    output["prob"] = output["prob"].astype("float32")
    output["abs_error"] = np.abs(output["prob"] - output["label"]).astype("float32")
    output["row_logloss"] = _row_logloss(output["label"].to_numpy(), output["prob"].to_numpy()).astype("float32")
    output = output.sort_values(["row_logloss", "abs_error"], ascending=[False, False]).reset_index(drop=True)
    output["is_hard"] = 0
    output["selection_group"] = "none"
    output["duplicate_count"] = 1

    default_duplicate_factor = max(1, int(default_duplicate_factor))
    class_duplicate_factors = class_duplicate_factors or {}

    if class_top_percents:
        for label, class_percent in class_top_percents.items():
            label = int(label)
            class_percent = float(class_percent)
            class_frame = output.loc[output["label"] == label].sort_values(
                ["row_logloss", "abs_error"],
                ascending=[False, False],
            )
            if class_frame.empty:
                continue
            top_k = max(1, int(math.ceil(len(class_frame) * class_percent)))
            selected_indices = class_frame.index[:top_k]
            output.loc[selected_indices, "is_hard"] = 1
            output.loc[selected_indices, "selection_group"] = f"label{label}"
            output.loc[selected_indices, "duplicate_count"] = max(
                1,
                int(class_duplicate_factors.get(label, default_duplicate_factor)),
            )
        return output.sort_values(["is_hard", "row_logloss", "abs_error"], ascending=[False, False, False]).reset_index(drop=True)

    top_percent = float(0.1 if top_percent is None else top_percent)
    top_k = max(1, int(math.ceil(len(output) * top_percent)))
    output.loc[: top_k - 1, "is_hard"] = 1
    output.loc[: top_k - 1, "selection_group"] = "global"
    output.loc[: top_k - 1, "duplicate_count"] = default_duplicate_factor
    return output


def expand_train_records_with_hard_examples(
    train_records: pd.DataFrame,
    hard_example_frame: pd.DataFrame,
    duplicate_factor: int | None = None,
) -> tuple[pd.DataFrame, int]:
    output = train_records.reset_index(drop=True).copy()
    hard_frame = hard_example_frame.loc[hard_example_frame["is_hard"] == 1].copy()
    if hard_frame.empty:
        return output, 0

    if "duplicate_count" not in hard_frame.columns:
        applied_duplicate_factor = max(1, int(1 if duplicate_factor is None else duplicate_factor))
        hard_frame["duplicate_count"] = applied_duplicate_factor

    duplicate_lookup: dict[str, int] = {
        str(row["id"]): max(1, int(row["duplicate_count"]))
        for _, row in hard_frame.iterrows()
    }
    expanded = [output]
    hard_count = 0

    for _, row in train_records.iterrows():
        sample_id = str(row["id"])
        sample_duplicate_count = duplicate_lookup.get(sample_id, 1)
        if sample_duplicate_count <= 1:
            continue
        hard_count += 1
        row_frame = pd.DataFrame([row.to_dict()])
        for _ in range(sample_duplicate_count - 1):
            expanded.append(row_frame.copy())

    return pd.concat(expanded, ignore_index=True), hard_count
