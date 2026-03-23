from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from .common import build_stratified_folds, ensure_output_dirs, load_records, write_json
from .hard_examples import build_hard_example_frame, expand_train_records_with_hard_examples
from .inference import predict_from_checkpoint
from .metrics import binary_logloss
from .seed import seed_everything
from .train import fit_model


def _aggregate_fold_predictions(frame: pd.DataFrame, with_labels: bool) -> pd.DataFrame:
    group_columns = ["id", "label"] if with_labels else ["id"]
    return (
        frame.groupby(group_columns, as_index=False)[["raw_logit", "prob"]]
        .mean()
        .sort_values("id")
        .reset_index(drop=True)
    )


def _get_oof_seeds(config: dict[str, Any]) -> list[int]:
    seeds = config["oof"].get("seeds")
    if seeds:
        return [int(seed) for seed in seeds]
    return [int(config["runtime"]["seed"])]


def _infer_reference_experiment_name(reference_oof_path: str | Path) -> str:
    stem = Path(reference_oof_path).stem
    return stem[:-4] if stem.endswith("_oof") else stem


def run_oof(config: dict[str, Any]) -> dict[str, Any]:
    output_dirs = ensure_output_dirs(config)
    experiment_name = str(config["experiment"]["name"])
    seeds = _get_oof_seeds(config)
    hard_ft_cfg = config.get("hard_finetune", {})
    hard_ft_enabled = bool(hard_ft_cfg.get("enabled", False))

    oof_frames: list[pd.DataFrame] = []
    dev_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    fold_scores: list[dict[str, Any]] = []
    manifest_frames: list[pd.DataFrame] = []

    train_records = load_records(config["paths"]["train_csv"])

    dev_records = None
    if bool(config["oof"].get("save_dev_predictions", True)):
        dev_records = load_records(config["paths"]["dev_csv"])
    test_records = (
        load_records(config["paths"]["sample_submission_csv"])[["id"]]
        if bool(config["oof"].get("save_test_predictions", True))
        else None
    )

    hard_example_frame = None
    hard_example_path = None
    if hard_ft_enabled:
        reference_oof_path = Path(str(hard_ft_cfg["reference_oof_path"]))
        if not reference_oof_path.exists():
            bootstrap_experiment_name = _infer_reference_experiment_name(reference_oof_path)
            bootstrap_config = copy.deepcopy(config)
            bootstrap_config["experiment"]["name"] = bootstrap_experiment_name
            bootstrap_config.pop("hard_finetune", None)
            print(
                f"[hardft] reference OOF not found: {reference_oof_path}. "
                f"bootstrapping plain OOF with experiment={bootstrap_experiment_name}"
            )
            run_oof(bootstrap_config)

        class_top_percents = None
        class_duplicate_factors = None
        if "label0_top_percent" in hard_ft_cfg or "label1_top_percent" in hard_ft_cfg:
            class_top_percents = {}
            if "label0_top_percent" in hard_ft_cfg:
                class_top_percents[0] = float(hard_ft_cfg["label0_top_percent"])
            if "label1_top_percent" in hard_ft_cfg:
                class_top_percents[1] = float(hard_ft_cfg["label1_top_percent"])
        if "label0_duplicate_factor" in hard_ft_cfg or "label1_duplicate_factor" in hard_ft_cfg:
            class_duplicate_factors = {}
            if "label0_duplicate_factor" in hard_ft_cfg:
                class_duplicate_factors[0] = int(hard_ft_cfg["label0_duplicate_factor"])
            if "label1_duplicate_factor" in hard_ft_cfg:
                class_duplicate_factors[1] = int(hard_ft_cfg["label1_duplicate_factor"])

        hard_example_frame = build_hard_example_frame(
            reference_oof_path=hard_ft_cfg["reference_oof_path"],
            top_percent=float(hard_ft_cfg.get("top_percent", 0.1)),
            class_top_percents=class_top_percents,
            class_duplicate_factors=class_duplicate_factors,
            default_duplicate_factor=int(hard_ft_cfg.get("duplicate_factor", 3)),
        )
        hard_example_path = output_dirs["logs"] / f"{experiment_name}_hard_examples.csv"
        hard_example_frame.to_csv(hard_example_path, index=False)
        if class_top_percents:
            print(
                f"[hardft] enabled reference={hard_ft_cfg['reference_oof_path']} "
                f"label0_top={float(class_top_percents.get(0, 0.0)):.2f} "
                f"label1_top={float(class_top_percents.get(1, 0.0)):.2f} "
                f"label0_dup={int((class_duplicate_factors or {}).get(0, hard_ft_cfg.get('duplicate_factor', 3)))} "
                f"label1_dup={int((class_duplicate_factors or {}).get(1, hard_ft_cfg.get('duplicate_factor', 3)))}"
            )
        else:
            print(
                f"[hardft] enabled reference={hard_ft_cfg['reference_oof_path']} "
                f"top_percent={float(hard_ft_cfg.get('top_percent', 0.1)):.2f} "
                f"duplicate_factor={int(hard_ft_cfg.get('duplicate_factor', 3))}"
            )

    for seed in seeds:
        seed_everything(seed)
        run_config = copy.deepcopy(config)
        run_config["runtime"]["seed"] = seed
        folds, fold_manifest = build_stratified_folds(
            train_records,
            n_splits=int(run_config["oof"]["n_splits"]),
            seed=seed,
        )
        fold_manifest["seed"] = seed
        manifest_frames.append(fold_manifest)

        for fold, (fold_train, fold_valid) in enumerate(folds):
            run_seed = seed * 100 + fold
            seed_everything(run_seed)
            run_config["runtime"]["seed"] = run_seed
            base_run_name = f"{experiment_name}_seed{seed}_fold{fold}"
            print(f"[oof] seed={seed} fold {fold + 1}/{len(folds)} run={base_run_name}")
            fit_result = fit_model(
                config=run_config,
                train_records=fold_train,
                valid_records=fold_valid,
                train_image_root=run_config["paths"]["train_image_root"],
                valid_image_root=run_config["paths"]["train_image_root"],
                run_name=base_run_name,
            )
            selected_result = fit_result
            hard_samples_in_train = 0

            if hard_ft_enabled and hard_example_frame is not None:
                expanded_train, hard_samples_in_train = expand_train_records_with_hard_examples(
                    train_records=fold_train,
                    hard_example_frame=hard_example_frame,
                    duplicate_factor=int(hard_ft_cfg.get("duplicate_factor", 3)),
                )
                finetune_run_config = copy.deepcopy(run_config)
                finetune_train_cfg = copy.deepcopy(finetune_run_config["train"])
                finetune_train_cfg["epochs"] = int(hard_ft_cfg.get("epochs", 3))
                finetune_train_cfg["lr"] = float(hard_ft_cfg.get("lr", finetune_train_cfg["lr"]))
                finetune_train_cfg["weight_decay"] = float(hard_ft_cfg.get("weight_decay", finetune_train_cfg["weight_decay"]))
                finetune_train_cfg["warmup_epochs"] = int(hard_ft_cfg.get("warmup_epochs", 0))
                finetune_train_cfg["patience"] = int(hard_ft_cfg.get("patience", 2))
                finetune_train_cfg["mixup_alpha"] = float(hard_ft_cfg.get("mixup_alpha", finetune_train_cfg.get("mixup_alpha", 0.0)))
                finetune_run_config["train"] = finetune_train_cfg
                finetune_run_name = f"{base_run_name}_hardft"
                print(
                    f"[hardft] run={finetune_run_name} hard_samples_in_train={hard_samples_in_train} "
                    f"expanded_rows={len(expanded_train)} lr={finetune_train_cfg['lr']:.1e} "
                    f"epochs={finetune_train_cfg['epochs']}"
                )
                selected_result = fit_model(
                    config=finetune_run_config,
                    train_records=expanded_train,
                    valid_records=fold_valid,
                    train_image_root=finetune_run_config["paths"]["train_image_root"],
                    valid_image_root=finetune_run_config["paths"]["train_image_root"],
                    run_name=finetune_run_name,
                    init_checkpoint_path=fit_result["best_checkpoint"],
                )

            valid_frame = pd.read_csv(selected_result["valid_prediction_path"])
            valid_frame["fold"] = fold
            valid_frame["seed"] = seed
            valid_frame["model_name"] = experiment_name
            oof_frames.append(valid_frame[["id", "label", "fold", "seed", "model_name", "raw_logit", "prob"]])

            fold_scores.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "run_name": selected_result["run_name"],
                    "best_logloss": selected_result["best_logloss"],
                    "best_checkpoint": selected_result["best_checkpoint"],
                    "base_best_logloss": fit_result["best_logloss"],
                    "hard_samples_in_train": hard_samples_in_train,
                }
            )

            checkpoint_path = selected_result["best_checkpoint"]
            if dev_records is not None:
                dev_frame = predict_from_checkpoint(
                    config=run_config,
                    checkpoint_path=checkpoint_path,
                    records=dev_records,
                    image_root=run_config["paths"]["dev_image_root"],
                    with_labels=True,
                )
                dev_frame["fold"] = fold
                dev_frame["seed"] = seed
                dev_frame["model_name"] = experiment_name
                dev_frames.append(dev_frame)

            if test_records is not None:
                test_frame = predict_from_checkpoint(
                    config=run_config,
                    checkpoint_path=checkpoint_path,
                    records=test_records,
                    image_root=run_config["paths"]["test_image_root"],
                    with_labels=False,
                )
                test_frame["fold"] = fold
                test_frame["seed"] = seed
                test_frame["model_name"] = experiment_name
                test_frames.append(test_frame)

    fold_manifest_path = output_dirs["oof"] / f"{experiment_name}_fold_manifest.csv"
    pd.concat(manifest_frames, ignore_index=True).to_csv(fold_manifest_path, index=False)

    oof_all = pd.concat(oof_frames, ignore_index=True)
    oof_all_path = output_dirs["oof"] / f"{experiment_name}_oof_all.csv"
    oof_all.to_csv(oof_all_path, index=False)
    oof_mean = _aggregate_fold_predictions(oof_all, with_labels=True)
    oof_path = output_dirs["oof"] / f"{experiment_name}_oof.csv"
    oof_mean.to_csv(oof_path, index=False)

    fold_scores_path = output_dirs["logs"] / f"{experiment_name}_fold_scores.json"
    write_json(fold_scores_path, fold_scores)
    oof_logloss = binary_logloss(oof_mean["label"].to_numpy(dtype="float32"), oof_mean["prob"].to_numpy(dtype="float32"))
    print(f"[oof] model={experiment_name} seeds={seeds} oof_logloss={oof_logloss:.6f}")

    results = {
        "oof_all_path": str(oof_all_path),
        "oof_path": str(oof_path),
        "fold_manifest_path": str(fold_manifest_path),
        "fold_scores_path": str(fold_scores_path),
        "oof_logloss": oof_logloss,
    }
    if hard_example_path is not None:
        results["hard_examples_path"] = str(hard_example_path)

    if dev_frames:
        dev_all = pd.concat(dev_frames, ignore_index=True)
        dev_all_path = output_dirs["oof"] / f"{experiment_name}_dev_fold_predictions.csv"
        dev_all.to_csv(dev_all_path, index=False)
        dev_mean = _aggregate_fold_predictions(dev_all, with_labels=True)
        dev_mean_path = output_dirs["oof"] / f"{experiment_name}_dev_mean.csv"
        dev_mean.to_csv(dev_mean_path, index=False)
        results["dev_fold_predictions_path"] = str(dev_all_path)
        results["dev_mean_path"] = str(dev_mean_path)

    if test_frames:
        test_all = pd.concat(test_frames, ignore_index=True)
        test_all_path = output_dirs["oof"] / f"{experiment_name}_test_fold_predictions.csv"
        test_all.to_csv(test_all_path, index=False)
        test_mean = _aggregate_fold_predictions(test_all, with_labels=False)
        test_mean_path = output_dirs["oof"] / f"{experiment_name}_test_mean.csv"
        test_mean.to_csv(test_mean_path, index=False)
        results["test_fold_predictions_path"] = str(test_all_path)
        results["test_mean_path"] = str(test_mean_path)

    return results
