from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import ensure_output_dirs, load_checkpoint, load_records, resolve_device
from .dual_view_dataset import DualViewDataset
from .dual_view_model import build_model_from_config
from .metrics import logits_to_unstable_probs
from .validate import predict_model


def get_eval_size_config(data_cfg: dict[str, Any]) -> tuple[int, int, int]:
    image_size = int(data_cfg["image_size"])
    resize_size = int(data_cfg.get("eval_resize_size", image_size))
    crop_size = int(data_cfg.get("eval_crop_size", image_size))
    return image_size, resize_size, crop_size


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_prediction_frame(ids: list[str], logits: np.ndarray, labels=None) -> pd.DataFrame:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    frame = pd.DataFrame(
        {
            "id": ids,
            "raw_logit": logits,
            "prob": logits_to_unstable_probs(logits),
        }
    )
    if labels is not None:
        frame["label"] = np.asarray(labels, dtype=np.int64).reshape(-1)
    return frame


def predict_from_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    records: pd.DataFrame,
    image_root: str | Path,
    *,
    with_labels: bool,
) -> pd.DataFrame:
    device = resolve_device(config["runtime"].get("device"))
    pin_memory = bool(config["runtime"]["pin_memory"] and device.type == "cuda")
    image_size, resize_size, crop_size = get_eval_size_config(config["data"])
    dataset = DualViewDataset(
        records=records,
        image_root=image_root,
        image_size=image_size,
        resize_size=resize_size,
        crop_size=crop_size,
        mean=list(config["data"]["mean"]),
        std=list(config["data"]["std"]),
        is_train=False,
        augmentation=None,
        with_labels=with_labels,
    )
    loader = build_loader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=pin_memory,
    )

    model = build_model_from_config(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location="cpu", strict=True)
    outputs = predict_model(model=model, loader=loader, device=device, use_amp=bool(config["runtime"]["amp"]))
    labels = records["label"].map(lambda x: 1 if str(x).lower() == "unstable" else 0).to_numpy() if with_labels else None
    return build_prediction_frame(outputs["ids"], outputs["logits"], labels=labels)


def predict_with_checkpoints_average(
    config: dict[str, Any],
    checkpoint_paths: list[str | Path],
    records: pd.DataFrame,
    image_root: str | Path,
    *,
    with_labels: bool,
) -> pd.DataFrame:
    frames = [predict_from_checkpoint(config, path, records, image_root, with_labels=with_labels) for path in checkpoint_paths]
    merged = frames[0][["id"]].copy()
    if with_labels:
        merged["label"] = frames[0]["label"]

    stacked_logits = np.stack([frame["raw_logit"].to_numpy(dtype=np.float32) for frame in frames], axis=0)
    merged["raw_logit"] = stacked_logits.mean(axis=0)
    merged["prob"] = logits_to_unstable_probs(merged["raw_logit"].to_numpy(dtype=np.float32))
    return merged


def make_submission(prediction_frame: pd.DataFrame, output_path: str | Path, clip_eps: float = 1e-4) -> Path:
    probs = np.clip(prediction_frame["prob"].to_numpy(dtype=np.float32), clip_eps, 1.0 - clip_eps)
    submission = pd.DataFrame(
        {
            "id": prediction_frame["id"],
            "unstable_prob": probs,
            "stable_prob": 1.0 - probs,
        }
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path


def run_single_model_inference(config: dict[str, Any]) -> dict[str, Any]:
    output_dirs = ensure_output_dirs(config)
    records = load_records(config["paths"]["sample_submission_csv"])[["id"]]
    prediction_frame = predict_from_checkpoint(
        config=config,
        checkpoint_path=config["inference"]["checkpoint_path"],
        records=records,
        image_root=config["paths"]["test_image_root"],
        with_labels=False,
    )
    submission_path = make_submission(
        prediction_frame,
        output_dirs["submissions"] / config["inference"]["submission_name"],
        clip_eps=float(config["inference"].get("clip_eps", 1e-4)),
    )
    print(f"[inference] saved submission to {submission_path}")
    return {
        "submission_path": str(submission_path),
        "num_rows": len(prediction_frame),
    }
