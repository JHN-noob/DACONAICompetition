from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .common import get_artifact_paths, to_project_relative
from .models import build_model, get_model_params, get_model_type


def _get_training_config(config: dict) -> dict:
    return config.get("training", {})


def _get_target_transform(config: dict) -> str:
    return str(_get_training_config(config).get("target_transform", "none")).lower()


def _get_sample_weight_mode(config: dict) -> str:
    return str(_get_training_config(config).get("sample_weight_mode", "none")).lower()


def _transform_target(values: pd.Series | np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if mode == "none":
        return array
    if mode == "log1p":
        return np.log1p(np.clip(array, a_min=0.0, a_max=None))
    raise ValueError(f"Unsupported target transform: {mode}")


def _inverse_transform_target(values: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if mode == "none":
        return array
    if mode == "log1p":
        return np.clip(np.expm1(array), a_min=0.0, a_max=None)
    raise ValueError(f"Unsupported target transform: {mode}")


def _fit_lightgbm(
    model,
    params: dict,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    categorical_columns: list[str],
    sample_weight_train: pd.Series | None = None,
    sample_weight_valid: pd.Series | None = None,
):
    import lightgbm as lgb

    callbacks = []
    eval_metric = params.get("metric", "l1")
    early_stopping_rounds = params.get("early_stopping_rounds")
    if early_stopping_rounds is not None:
        callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False))

    verbose_eval = params.get("verbose_eval")
    if verbose_eval is not None:
        callbacks.append(lgb.log_evaluation(period=int(verbose_eval)))

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_valid, y_valid)],
        "eval_metric": eval_metric,
        "callbacks": callbacks,
    }
    if categorical_columns:
        fit_kwargs["categorical_feature"] = categorical_columns
    if sample_weight_train is not None:
        fit_kwargs["sample_weight"] = sample_weight_train.to_numpy(dtype=np.float64)
    if sample_weight_valid is not None:
        fit_kwargs["eval_sample_weight"] = [sample_weight_valid.to_numpy(dtype=np.float64)]

    model.fit(**fit_kwargs)
    best_iteration = model.best_iteration_ or model.n_estimators
    pred_valid = model.predict(X_valid, num_iteration=best_iteration)
    pred_test = model.predict(X_test, num_iteration=best_iteration)
    return pred_valid, pred_test, best_iteration


def _extract_feature_importance(model, feature_columns: list[str], fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": feature_columns,
            "fold": fold,
            "importance": model.feature_importances_,
        }
    )


def _save_model(model, model_dir: Path, run_name: str, fold: int) -> str:
    path = model_dir / f"{run_name}_fold{fold}.txt"
    model.booster_.save_model(str(path))
    return to_project_relative(path)


def train_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    config: dict,
    sample_weights: pd.Series | None = None,
    extra_metrics: dict | None = None,
    fit_target_values: pd.Series | None = None,
    prediction_offset_train: pd.Series | None = None,
    prediction_offset_test: pd.Series | None = None,
) -> dict:
    id_col = config["columns"]["id_col"]
    group_col = config["columns"]["group_col"]
    target_col = config["columns"]["target_col"]
    n_splits = int(config["cv"]["n_splits"])
    run_name = config["run_name"]
    model_type = get_model_type(config)
    if model_type != "lightgbm":
        raise ValueError(f"Only LightGBM is supported in the final pipeline: {model_type}")

    model_params = get_model_params(config)
    cv_strategy = str(config["cv"].get("strategy", "scenario_stratified")).lower()
    target_transform = _get_target_transform(config)
    sample_weight_mode = _get_sample_weight_mode(config)
    artifacts = get_artifact_paths(config)

    train = train_df.merge(
        fold_df[[id_col, "fold"]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )
    if train["fold"].isna().any():
        raise ValueError("Some training rows are missing fold assignments.")

    test = test_df.copy()

    if fit_target_values is not None:
        fit_target_frame = fit_target_values.rename("__fit_target").rename_axis(id_col).reset_index()
        train = train.merge(fit_target_frame, on=id_col, how="left", validate="one_to_one")
        if train["__fit_target"].isna().any():
            raise ValueError("Some training rows are missing fit_target values.")

    if prediction_offset_train is not None:
        offset_train_frame = prediction_offset_train.rename("__prediction_offset").rename_axis(id_col).reset_index()
        train = train.merge(offset_train_frame, on=id_col, how="left", validate="one_to_one")
        if train["__prediction_offset"].isna().any():
            raise ValueError("Some training rows are missing prediction offsets.")

    if prediction_offset_test is not None:
        offset_test_frame = prediction_offset_test.rename("__prediction_offset").rename_axis(id_col).reset_index()
        test = test.merge(offset_test_frame, on=id_col, how="left", validate="one_to_one")
        if test["__prediction_offset"].isna().any():
            raise ValueError("Some test rows are missing prediction offsets.")

    sample_weight_summary = None
    if sample_weights is not None:
        weight_frame = sample_weights.rename("__sample_weight").rename_axis(id_col).reset_index()
        train = train.merge(weight_frame, on=id_col, how="left", validate="one_to_one")
        if train["__sample_weight"].isna().any():
            raise ValueError("Some training rows are missing sample weights.")
        sample_weight_summary = {
            "min": float(train["__sample_weight"].min()),
            "max": float(train["__sample_weight"].max()),
            "mean": float(train["__sample_weight"].mean()),
            "std": float(train["__sample_weight"].std(ddof=0)),
        }

    oof_predictions = np.zeros(len(train), dtype=np.float32)
    test_predictions = np.zeros(len(test), dtype=np.float64)
    fold_metrics: list[dict] = []
    importance_frames: list[pd.DataFrame] = []
    X_test_full = test[feature_columns]
    prediction_clip_min = _get_training_config(config).get("prediction_clip_min")
    if prediction_clip_min is not None:
        prediction_clip_min = float(prediction_clip_min)

    for fold in range(n_splits):
        train_mask = train["fold"] != fold
        valid_mask = train["fold"] == fold

        X_train = train.loc[train_mask, feature_columns]
        X_valid = train.loc[valid_mask, feature_columns]
        if "__fit_target" in train.columns:
            y_train_raw = train.loc[train_mask, "__fit_target"]
            y_valid_fit_raw = train.loc[valid_mask, "__fit_target"]
        else:
            y_train_raw = train.loc[train_mask, target_col]
            y_valid_fit_raw = train.loc[valid_mask, target_col]
        y_valid_metric_raw = train.loc[valid_mask, target_col]

        sample_weight_train = train.loc[train_mask, "__sample_weight"] if "__sample_weight" in train.columns else None
        sample_weight_valid = train.loc[valid_mask, "__sample_weight"] if "__sample_weight" in train.columns else None
        y_train = _transform_target(y_train_raw, target_transform)
        y_valid = _transform_target(y_valid_fit_raw, target_transform)

        model = build_model(config, params_override=model_params)
        pred_valid_raw, pred_test_raw, best_iteration = _fit_lightgbm(
            model,
            model_params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test_full,
            categorical_columns,
            sample_weight_train=sample_weight_train,
            sample_weight_valid=sample_weight_valid,
        )

        pred_valid = _inverse_transform_target(pred_valid_raw, target_transform)
        pred_test = _inverse_transform_target(pred_test_raw, target_transform)
        if "__prediction_offset" in train.columns:
            pred_valid = pred_valid + train.loc[valid_mask, "__prediction_offset"].to_numpy(dtype=np.float64)
        if "__prediction_offset" in test.columns:
            pred_test = pred_test + test["__prediction_offset"].to_numpy(dtype=np.float64)
        if prediction_clip_min is not None:
            pred_valid = np.clip(pred_valid, a_min=prediction_clip_min, a_max=None)
            pred_test = np.clip(pred_test, a_min=prediction_clip_min, a_max=None)

        fold_mae = float(mean_absolute_error(y_valid_metric_raw, pred_valid))
        oof_predictions[valid_mask.to_numpy()] = np.asarray(pred_valid, dtype=np.float32)
        test_predictions += np.asarray(pred_test, dtype=np.float64) / n_splits
        model_path = _save_model(model, artifacts["model_dir"], run_name, fold)
        importance_frames.append(_extract_feature_importance(model, feature_columns, fold))
        fold_metrics.append(
            {
                "fold": fold,
                "valid_mae": fold_mae,
                "best_iteration": int(best_iteration) if best_iteration is not None else None,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "model_path": model_path,
            }
        )

        print(
            f"[cv] run={run_name} fold={fold} "
            f"train_rows={int(train_mask.sum())} valid_rows={int(valid_mask.sum())} "
            f"valid_mae={fold_mae:.6f}"
        )

    oof_mae = float(mean_absolute_error(train[target_col], oof_predictions))
    oof_frame = train[[id_col, group_col, "fold", target_col]].copy()
    oof_frame["prediction"] = oof_predictions
    oof_frame.to_csv(artifacts["oof_path"], index=False, encoding="utf-8-sig")

    test_frame = test[[id_col]].copy()
    test_frame["prediction"] = np.asarray(test_predictions, dtype=np.float32)
    test_frame.to_csv(artifacts["test_pred_path"], index=False, encoding="utf-8-sig")

    importance_frame = pd.concat(importance_frames, ignore_index=True)
    importance_summary = (
        importance_frame.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
        .sort_values("importance_mean", ascending=False)
    )
    importance_summary.to_csv(artifacts["importance_path"], index=False, encoding="utf-8-sig")

    metrics = {
        "run_name": run_name,
        "model_type": model_type,
        "cv_strategy": cv_strategy,
        "feature_count": len(feature_columns),
        "categorical_columns": categorical_columns,
        "target_transform": target_transform,
        "sample_weight_mode": sample_weight_mode,
        "fold_metrics": fold_metrics,
        "oof_mae": oof_mae,
    }
    if sample_weight_summary is not None:
        metrics["sample_weight_summary"] = sample_weight_summary
    if sample_weights is not None and sample_weights.attrs.get("domain_metrics") is not None:
        metrics["domain_adaptation"] = sample_weights.attrs["domain_metrics"]
    if extra_metrics:
        metrics.update(extra_metrics)

    with artifacts["metrics_path"].open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"[cv] run={run_name} oof_mae={oof_mae:.6f}")
    return metrics
