from __future__ import annotations

import torch
import torch.nn as nn

from .heads import MLPHead, ViewAdapter


def _import_timm():
    try:
        import timm
    except ImportError as error:
        raise ImportError("timm is required to build the backbone. Install it in the notebook environment.") from error
    return timm


class DualViewModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        pretrained: bool,
        head_hidden_dim: int,
        drop_rate: float,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        timm = _import_timm()
        try:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                img_size=image_size,
            )
        except TypeError:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )

        feature_dim = getattr(self.encoder, "num_features", None)
        if feature_dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, image_size, image_size)
                feature_dim = int(self.encoder(dummy).shape[-1])
        self.front_adapter = ViewAdapter(feature_dim, drop_rate, activation=activation)
        self.top_adapter = ViewAdapter(feature_dim, drop_rate, activation=activation)
        self.head = MLPHead(feature_dim * 2, head_hidden_dim, drop_rate, activation=activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        front_features = self.front_adapter(self.encode(front))
        top_features = self.top_adapter(self.encode(top))
        fused = torch.cat([front_features, top_features], dim=1)
        return self.head(fused).squeeze(1)


def build_model_from_config(config: dict) -> DualViewModel:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return DualViewModel(
        backbone_name=model_cfg["backbone_name"],
        image_size=int(data_cfg["image_size"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", 512)),
        drop_rate=float(model_cfg.get("drop_rate", 0.3)),
        activation=str(model_cfg.get("activation", "gelu")),
    )
