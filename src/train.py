from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import (
    AverageMeter,
    autocast_context,
    create_grad_scaler,
    ensure_output_dirs,
    load_checkpoint,
    load_records,
    resolve_device,
    save_checkpoint,
    split_train_valid,
    write_history,
)
from .dual_view_dataset import DualViewDataset
from .dual_view_model import build_model_from_config
from .losses import build_loss
from .metrics import logits_to_unstable_probs
from .seed import seed_everything
from .validate import evaluate_model


def get_split_size_config(data_cfg: dict[str, Any], is_train: bool) -> tuple[int, int, int]:
    image_size = int(data_cfg["image_size"])
    resize_size = int(data_cfg.get("train_resize_size" if is_train else "eval_resize_size", image_size))
    crop_size = int(data_cfg.get("train_crop_size" if is_train else "eval_crop_size", image_size))
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


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int):
    warmup_epochs = max(0, min(int(warmup_epochs), int(epochs) - 1))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def apply_mixup(
    front: torch.Tensor,
    top: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = float(alpha)
    if alpha <= 0.0 or labels.size(0) < 2:
        return front, top, labels

    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(labels.size(0), device=labels.device)
    mixed_front = lam * front + (1.0 - lam) * front[indices]
    mixed_top = lam * top + (1.0 - lam) * top[indices]
    mixed_labels = lam * labels + (1.0 - lam) * labels[indices]
    return mixed_front, mixed_top, mixed_labels


def build_prediction_frame(ids: list[str], logits: torch.Tensor | list[float] | Any, labels=None) -> pd.DataFrame:
    logits_np = torch.as_tensor(logits).detach().cpu().numpy() if torch.is_tensor(logits) else pd.Series(logits).to_numpy()
    logits_np = logits_np.astype("float32", copy=False).reshape(-1)
    probs = logits_to_unstable_probs(logits_np)
    frame = pd.DataFrame(
        {
            "id": ids,
            "raw_logit": logits_np,
            "prob": probs,
        }
    )
    if labels is not None:
        label_np = torch.as_tensor(labels).detach().cpu().numpy() if torch.is_tensor(labels) else pd.Series(labels).to_numpy()
        frame["label"] = label_np.astype("int64", copy=False)
    return frame


def get_default_split_from_config(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    train_records = load_records(config["paths"]["train_csv"])
    use_dev_for_validation = bool(config["data"].get("use_dev_for_validation", True))
    if use_dev_for_validation:
        valid_records = load_records(config["paths"]["dev_csv"])
        return (
            train_records,
            valid_records,
            config["paths"]["train_image_root"],
            config["paths"]["dev_image_root"],
        )

    split_train, split_valid = split_train_valid(
        train_records,
        val_ratio=float(config["data"].get("val_ratio", 0.1)),
        seed=int(config["runtime"]["seed"]),
    )
    return (
        split_train,
        split_valid,
        config["paths"]["train_image_root"],
        config["paths"]["train_image_root"],
    )


def fit_model(
    config: dict[str, Any],
    train_records: pd.DataFrame,
    valid_records: pd.DataFrame,
    train_image_root: str | Path,
    valid_image_root: str | Path,
    run_name: str,
    init_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dirs = ensure_output_dirs(config)
    device = resolve_device(config["runtime"].get("device"))
    pin_memory = bool(config["runtime"]["pin_memory"] and device.type == "cuda")
    use_amp = bool(config["runtime"]["amp"])

    data_cfg = config["data"]
    train_cfg = config["train"]
    train_image_size, train_resize_size, train_crop_size = get_split_size_config(data_cfg, is_train=True)
    eval_image_size, eval_resize_size, eval_crop_size = get_split_size_config(data_cfg, is_train=False)

    train_dataset = DualViewDataset(
        records=train_records,
        image_root=train_image_root,
        image_size=train_image_size,
        resize_size=train_resize_size,
        crop_size=train_crop_size,
        mean=list(data_cfg["mean"]),
        std=list(data_cfg["std"]),
        is_train=True,
        augmentation=config.get("augmentation", {}),
        with_labels=True,
    )
    valid_dataset = DualViewDataset(
        records=valid_records,
        image_root=valid_image_root,
        image_size=eval_image_size,
        resize_size=eval_resize_size,
        crop_size=eval_crop_size,
        mean=list(data_cfg["mean"]),
        std=list(data_cfg["std"]),
        is_train=False,
        augmentation=None,
        with_labels=True,
    )

    train_loader = build_loader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=pin_memory,
    )
    valid_loader = build_loader(
        valid_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=pin_memory,
    )

    model = build_model_from_config(config).to(device)
    if init_checkpoint_path is not None:
        load_checkpoint(init_checkpoint_path, model=model, map_location="cpu", strict=True)
    criterion = build_loss(config)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, int(train_cfg["epochs"]), int(train_cfg.get("warmup_epochs", 0)))
    scaler = create_grad_scaler(use_amp)
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))

    best_checkpoint_path = output_dirs["checkpoints"] / f"{run_name}_best.pth"
    last_checkpoint_path = output_dirs["checkpoints"] / f"{run_name}_last.pth"
    history_path = output_dirs["logs"] / f"{run_name}_history.csv"
    valid_prediction_path = output_dirs["logs"] / f"{run_name}_valid_predictions.csv"

    best_logloss = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []
    best_valid_frame: pd.DataFrame | None = None

    print(
        f"[train] run={run_name} backbone={config['model']['backbone_name']} "
        f"batch_size={train_cfg['batch_size']} lr={train_cfg['lr']:.1e} "
        f"label_smoothing={float(train_cfg.get('label_smoothing', 0.0)):.3f} mixup_alpha={mixup_alpha:.3f} "
        f"init_checkpoint={'yes' if init_checkpoint_path is not None else 'no'} "
        f"train_resize={train_resize_size} train_crop={train_crop_size} "
        f"eval_resize={eval_resize_size} eval_crop={eval_crop_size}"
    )

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        train_loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, leave=False, desc=f"{run_name} epoch {epoch}"), start=1):
            front = batch["front"].to(device, non_blocking=True)
            top = batch["top"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            front, top, labels = apply_mixup(front, top, labels, mixup_alpha)

            with autocast_context(device, use_amp):
                logits = model(front, top)
                loss = criterion(logits, labels)
                scaled_loss = loss / int(train_cfg["grad_accum_steps"])

            scaler.scale(scaled_loss).backward()

            should_step = step % int(train_cfg["grad_accum_steps"]) == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_meter.update(loss.item(), front.size(0))

        scheduler.step()
        valid_metrics = evaluate_model(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss_meter.avg, 6),
                "valid_loss": round(valid_metrics["loss"], 6),
                "valid_logloss": round(valid_metrics["logloss"], 6),
                "valid_accuracy": round(valid_metrics["accuracy"], 6),
                "valid_auc": round(valid_metrics["auc"], 6) if valid_metrics["auc"] == valid_metrics["auc"] else None,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[train] {run_name} epoch={epoch}/{train_cfg['epochs']} "
            f"train_loss={train_loss_meter.avg:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_logloss={valid_metrics['logloss']:.4f} valid_acc={valid_metrics['accuracy']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_logloss,
            extra={"run_name": run_name},
        )

        if valid_metrics["logloss"] < best_logloss:
            best_logloss = float(valid_metrics["logloss"])
            patience_counter = 0
            best_valid_frame = build_prediction_frame(
                ids=valid_metrics["ids"],
                logits=valid_metrics["logits"],
                labels=valid_metrics["labels"],
            )
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_logloss,
                extra={"run_name": run_name},
            )
        else:
            patience_counter += 1

        if int(train_cfg["patience"]) > 0 and patience_counter >= int(train_cfg["patience"]):
            print(f"[train] early stop run={run_name} epoch={epoch}")
            break

    write_history(history_path, history)
    if best_valid_frame is not None:
        best_valid_frame.to_csv(valid_prediction_path, index=False)

    return {
        "run_name": run_name,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "history_path": str(history_path),
        "valid_prediction_path": str(valid_prediction_path),
        "best_logloss": best_logloss,
    }


def run_single_model_experiment(config: dict[str, Any]) -> dict[str, Any]:
    seed_everything(int(config["runtime"]["seed"]))
    train_records, valid_records, train_root, valid_root = get_default_split_from_config(config)
    return fit_model(
        config=config,
        train_records=train_records,
        valid_records=valid_records,
        train_image_root=train_root,
        valid_image_root=valid_root,
        run_name=str(config["experiment"]["name"]),
    )
