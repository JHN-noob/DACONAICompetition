from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, smoothing: float = 0.0) -> None:
        super().__init__()
        self.smoothing = float(max(0.0, min(smoothing, 0.499)))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        targets = targets.float()
        if self.smoothing > 0.0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets)

def build_loss(config_or_train_cfg: dict[str, Any] | None = None) -> nn.Module:
    train_cfg: dict[str, Any] = {}
    if config_or_train_cfg is not None:
        train_cfg = config_or_train_cfg["train"] if "train" in config_or_train_cfg else config_or_train_cfg
    smoothing = float(train_cfg.get("label_smoothing", 0.0))
    return SmoothedBCEWithLogitsLoss(smoothing=smoothing)
