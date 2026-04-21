from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from .common import parse_label


def apply_jpeg_compression(image: Image.Image, quality: int | None) -> Image.Image:
    if quality is None:
        return image
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")
    compressed.load()
    return compressed


def apply_shadow(
    image: Image.Image,
    polygon: list[tuple[int, int]] | None,
    strength: float | None,
) -> Image.Image:
    if polygon is None or strength is None:
        return image

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)

    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    image_np = np.asarray(image, dtype=np.float32)
    shadowed = image_np * (1.0 - float(strength) * mask_np[..., None])
    return Image.fromarray(np.clip(shadowed, 0.0, 255.0).astype(np.uint8))


def apply_gaussian_noise(tensor: torch.Tensor, std: float | None) -> torch.Tensor:
    if std is None or std <= 0.0:
        return tensor
    noise = torch.randn_like(tensor) * float(std)
    return torch.clamp(tensor + noise, 0.0, 1.0)


class DualViewDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        image_root: str | Path,
        image_size: int,
        resize_size: int | None,
        crop_size: int | None,
        mean: list[float],
        std: list[float],
        *,
        is_train: bool,
        augmentation: dict[str, Any] | None = None,
        with_labels: bool = True,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.image_size = int(image_size)
        self.resize_size = int(resize_size if resize_size is not None else image_size)
        self.crop_size = int(crop_size if crop_size is not None else image_size)
        self.crop_size = min(self.crop_size, self.resize_size)
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.augmentation = augmentation or {}
        self.with_labels = with_labels and "label" in self.records.columns

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _resize(self, image: Image.Image) -> Image.Image:
        return F.resize(image, [self.resize_size, self.resize_size], interpolation=InterpolationMode.BILINEAR, antialias=True)

    def _sample_crop_box(self, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self.crop_size >= self.resize_size:
            return None
        width, height = size
        max_top = max(height - self.crop_size, 0)
        max_left = max(width - self.crop_size, 0)
        top = random.randint(0, max_top) if self.is_train else max_top // 2
        left = random.randint(0, max_left) if self.is_train else max_left // 2
        return top, left, self.crop_size, self.crop_size

    def _apply_crop(self, image: Image.Image, crop_box: tuple[int, int, int, int] | None) -> Image.Image:
        if crop_box is None:
            return image
        top, left, height, width = crop_box
        return F.crop(image, top=top, left=left, height=height, width=width)

    def _sample_value(self, key: str, default: float) -> float:
        value = self.augmentation.get(key, default)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return random.uniform(float(value[0]), float(value[1]))
        return float(value)

    def _sample_affine_params(self, size: tuple[int, int]) -> dict[str, Any] | None:
        if not self.is_train or random.random() >= float(self.augmentation.get("affine_p", 0.0)):
            return None

        width, height = size
        shift_limit = float(self.augmentation.get("shift_limit", 0.0))
        scale_limit = float(self.augmentation.get("scale_limit", 0.0))
        rotate_limit = float(self.augmentation.get("rotate_limit", 0.0))
        return {
            "translate": (
                int(round(random.uniform(-shift_limit, shift_limit) * width)),
                int(round(random.uniform(-shift_limit, shift_limit) * height)),
            ),
            "scale": random.uniform(1.0 - scale_limit, 1.0 + scale_limit),
            "angle": random.uniform(-rotate_limit, rotate_limit),
        }

    def _sample_shadow_plan(self, size: tuple[int, int]) -> tuple[list[tuple[int, int]] | None, float | None]:
        if not self.is_train or random.random() >= self._sample_value("shadow_p", 0.0):
            return None, None

        width, height = size
        num_vertices = random.randint(3, 5)
        polygon = [
            (
                random.randint(0, max(width - 1, 0)),
                random.randint(0, max(height - 1, 0)),
            )
            for _ in range(num_vertices)
        ]
        strength = random.uniform(0.25, 0.45)
        return polygon, strength

    def _sample_train_plan(self, size: tuple[int, int]) -> dict[str, Any]:
        if not self.is_train:
            return {"affine": None}

        plan: dict[str, Any] = {
            "crop_box": self._sample_crop_box(size),
            "affine": self._sample_affine_params(size),
            "brightness": None,
            "contrast": None,
            "gamma": None,
            "shadow_polygon": None,
            "shadow_strength": None,
            "compression_quality": None,
            "blur_sigma": None,
            "noise_std": None,
        }

        if random.random() < float(self.augmentation.get("brightness_contrast_p", 0.0)):
            brightness_limit = self._sample_value("brightness_limit", 0.2)
            contrast_limit = self._sample_value("contrast_limit", 0.2)
            plan["brightness"] = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            plan["contrast"] = 1.0 + random.uniform(-contrast_limit, contrast_limit)

        if random.random() < float(self.augmentation.get("gamma_p", 0.0)):
            plan["gamma"] = random.uniform(*tuple(self.augmentation.get("gamma_range", [0.9, 1.1])))

        shadow_polygon, shadow_strength = self._sample_shadow_plan(size)
        plan["shadow_polygon"] = shadow_polygon
        plan["shadow_strength"] = shadow_strength

        if random.random() < float(self.augmentation.get("compression_p", 0.0)):
            quality_range = tuple(self.augmentation.get("jpeg_quality_range", [50, 95]))
            plan["compression_quality"] = random.randint(int(quality_range[0]), int(quality_range[1]))

        if random.random() < float(self.augmentation.get("blur_p", 0.0)):
            blur_sigma_range = tuple(self.augmentation.get("blur_sigma_range", [0.1, 1.5]))
            plan["blur_sigma"] = random.uniform(float(blur_sigma_range[0]), float(blur_sigma_range[1]))

        if random.random() < float(self.augmentation.get("noise_p", 0.0)):
            noise_std_range = tuple(self.augmentation.get("noise_std_range", [0.0, 0.03]))
            plan["noise_std"] = random.uniform(float(noise_std_range[0]), float(noise_std_range[1]))

        return plan

    def _apply_affine(self, image: Image.Image, affine_params: dict[str, Any] | None) -> Image.Image:
        if affine_params is None:
            return image
        return F.affine(
            image,
            angle=float(affine_params["angle"]),
            translate=tuple(int(value) for value in affine_params["translate"]),
            scale=float(affine_params["scale"]),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=[0, 0, 0],
        )

    def _apply_train_augmentation(self, image: Image.Image, plan: dict[str, Any]) -> Image.Image:
        if not self.is_train:
            return image

        image = self._apply_crop(image, plan.get("crop_box"))
        image = self._apply_affine(image, plan.get("affine"))
        if plan.get("brightness") is not None:
            image = F.adjust_brightness(image, float(plan["brightness"]))
        if plan.get("contrast") is not None:
            image = F.adjust_contrast(image, float(plan["contrast"]))
        if plan.get("gamma") is not None:
            image = F.adjust_gamma(image, gamma=float(plan["gamma"]))
        image = apply_shadow(
            image,
            polygon=plan.get("shadow_polygon"),
            strength=plan.get("shadow_strength"),
        )
        image = apply_jpeg_compression(image, quality=plan.get("compression_quality"))
        if plan.get("blur_sigma") is not None:
            blur_sigma = float(plan["blur_sigma"])
            image = F.gaussian_blur(image, kernel_size=5, sigma=[blur_sigma, blur_sigma])
        return image

    def _to_tensor(self, image: Image.Image, noise_std: float | None = None) -> torch.Tensor:
        tensor = F.pil_to_tensor(image).float() / 255.0
        tensor = apply_gaussian_noise(tensor, noise_std)
        tensor = F.normalize(tensor, mean=self.mean, std=self.std)
        return tensor

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.records.iloc[index]
        sample_id = str(row["id"])
        row_image_root = row.get("image_root") if hasattr(row, "get") else None
        sample_root = Path(row_image_root) if row_image_root not in (None, "", "nan") else self.image_root
        sample_dir = sample_root / sample_id

        front_image = self._resize(self._load_image(sample_dir / "front.png"))
        top_image = self._resize(self._load_image(sample_dir / "top.png"))
        if self.is_train:
            train_plan = self._sample_train_plan(front_image.size)
            front = self._to_tensor(
                self._apply_train_augmentation(front_image, train_plan),
                noise_std=train_plan.get("noise_std"),
            )
            top = self._to_tensor(
                self._apply_train_augmentation(top_image, train_plan),
                noise_std=train_plan.get("noise_std"),
            )
        else:
            eval_crop_box = self._sample_crop_box(front_image.size)
            front = self._to_tensor(self._apply_crop(front_image, eval_crop_box))
            top = self._to_tensor(self._apply_crop(top_image, eval_crop_box))

        batch: dict[str, torch.Tensor | str] = {
            "id": sample_id,
            "front": front,
            "top": top,
        }
        if self.with_labels:
            batch["label"] = torch.tensor(parse_label(row["label"]), dtype=torch.float32)
        return batch
