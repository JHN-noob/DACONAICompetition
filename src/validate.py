from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .common import AverageMeter, autocast_context
from .metrics import summarize_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    loss_meter = AverageMeter()
    logits_list = []
    labels_list = []
    ids: list[str] = []

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            logits = model(front, top)
            loss = criterion(logits, labels)

        loss_meter.update(loss.item(), front.size(0))
        logits_list.append(logits.detach().float().cpu().numpy().astype(np.float32, copy=False))
        labels_list.append(labels.detach().float().cpu().numpy().astype(np.float32, copy=False))
        ids.extend(batch["id"])

    logits_np = np.concatenate(logits_list)
    labels_np = np.concatenate(labels_list)
    metrics = summarize_metrics(labels_np, logits_np)
    metrics["loss"] = loss_meter.avg
    metrics["ids"] = ids
    metrics["logits"] = logits_np
    metrics["labels"] = labels_np
    return metrics


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    logits_list = []
    ids: list[str] = []

    for batch in loader:
        front = batch["front"].to(device, non_blocking=True)
        top = batch["top"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(front, top)
        logits_list.append(logits.detach().float().cpu().numpy().astype(np.float32, copy=False))
        ids.extend(batch["id"])

    return {
        "ids": ids,
        "logits": np.concatenate(logits_list),
    }
