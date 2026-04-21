from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .common import load_config, prepare_runtime, resolve_path
from .data_io import load_data
from .features import build_features
from .folds import make_cv_folds
from .inference import save_submission
from .schema import validate_stacking_schema
from .train import train_cv


def _load_member_predictions(config: dict, member_run_name: str, split: str) -> pd.DataFrame:
    output_dir = resolve_path(config["paths"]["output_dir"])
    path = output_dir / "oof" / f"{member_run_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing stacking member artifact: {path}. "
            "Run the required upstream config first to create OOF/test predictions."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_member_artifacts(config: dict, member_run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        _load_member_predictions(config, member_run_name, "oof"),
        _load_member_predictions(config, member_run_name, "test"),
    )


def _get_stacking_usecols_map(config: dict) -> dict[str, list[str]]:
    id_col = config["columns"]["id_col"]
    group_col = config["columns"]["group_col"]
    target_col = config["columns"]["target_col"]
    meta_feature_groups = set(config["stacking"].get("meta_features", []))

    layout_columns = ["layout_id"]
    if "layout_type" in meta_feature_groups:
        layout_columns.append("layout_type")

    return {
        "train": [id_col, group_col, target_col, "layout_id"],
        "test": [id_col, group_col, "layout_id"],
        "layout": layout_columns,
        "sub": [id_col, target_col],
    }


def _load_context_frames(
    config: dict,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stacking_config = config["stacking"]
    context_columns = stacking_config.get("context_columns", [])
    if not context_columns:
        return pd.DataFrame(columns=[id_col]), pd.DataFrame(columns=[id_col])

    source_config_path = stacking_config.get("context_source_config")
    if not source_config_path:
        raise ValueError("stacking.context_columns is set but stacking.context_source_config is missing.")

    source_config = load_config(source_config_path)
    source_data = load_data(source_config)
    train_feature_frame, test_feature_frame, _, _ = build_features(
        train_df=source_data["train"],
        test_df=source_data["test"],
        layout_df=source_data["layout"],
        config=source_config,
    )

    missing_columns = [column for column in context_columns if column not in train_feature_frame.columns]
    if missing_columns:
        raise ValueError(
            "Some stacking.context_columns are missing from the context source feature frame: "
            f"{missing_columns}"
        )

    renamed_columns = {column: f"ctx__{column}" for column in context_columns}
    train_context = train_feature_frame[[id_col, *context_columns]].rename(columns=renamed_columns)
    test_context = test_feature_frame[[id_col, *context_columns]].rename(columns=renamed_columns)
    return train_context, test_context


def _add_prediction_stats(frame: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    values = frame[prediction_columns]
    frame["stack_pred_mean"] = values.mean(axis=1).astype("float32")
    frame["stack_pred_std"] = values.std(axis=1).astype("float32")
    frame["stack_pred_min"] = values.min(axis=1).astype("float32")
    frame["stack_pred_max"] = values.max(axis=1).astype("float32")
    return frame


def _add_pairwise_diff(frame: pd.DataFrame, prediction_columns: list[str]) -> pd.DataFrame:
    for idx, left_column in enumerate(prediction_columns):
        for right_column in prediction_columns[idx + 1 :]:
            diff_name = f"diff__{left_column}__{right_column}"
            frame[diff_name] = (frame[left_column] - frame[right_column]).astype("float32")
    return frame


def _build_meta_frames(
    data: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], dict]:
    id_col = config["columns"]["id_col"]
    group_col = config["columns"]["group_col"]
    target_col = config["columns"]["target_col"]
    stacking_config = config["stacking"]
    members = stacking_config["members"]
    meta_feature_groups = set(stacking_config.get("meta_features", []))

    train_meta = data["train"][[id_col, group_col, target_col, "layout_id"]].copy()
    test_meta = data["test"][[id_col, group_col, "layout_id"]].copy()

    if "layout_type" in meta_feature_groups and "layout_type" in data["layout"].columns:
        layout_type_frame = data["layout"][["layout_id", "layout_type"]].copy()
        train_meta = train_meta.merge(layout_type_frame, on="layout_id", how="left", validate="many_to_one")
        test_meta = test_meta.merge(layout_type_frame, on="layout_id", how="left", validate="many_to_one")
        train_meta["layout_type"] = train_meta["layout_type"].fillna("__MISSING__").astype("category")
        test_meta["layout_type"] = test_meta["layout_type"].fillna("__MISSING__").astype("category")

    prediction_columns: list[str] = []
    for member in members:
        column_name = f"pred__{member}"
        train_member, test_member = _load_member_artifacts(config, member)
        train_meta = train_meta.merge(
            train_member[[id_col, "prediction"]].rename(columns={"prediction": column_name}),
            on=id_col,
            how="left",
            validate="one_to_one",
        )
        test_meta = test_meta.merge(
            test_member[[id_col, "prediction"]].rename(columns={"prediction": column_name}),
            on=id_col,
            how="left",
            validate="one_to_one",
        )
        prediction_columns.append(column_name)

    train_context, test_context = _load_context_frames(config=config, id_col=id_col)
    if len(train_context.columns) > 1:
        train_meta = train_meta.merge(train_context, on=id_col, how="left", validate="one_to_one")
        test_meta = test_meta.merge(test_context, on=id_col, how="left", validate="one_to_one")

    if train_meta[prediction_columns].isna().any().any() or test_meta[prediction_columns].isna().any().any():
        raise ValueError("Some stacking member predictions are missing.")

    for frame in (train_meta, test_meta):
        frame[prediction_columns] = frame[prediction_columns].astype("float32")
        if "prediction_stats" in meta_feature_groups:
            _add_prediction_stats(frame, prediction_columns)
        if "pairwise_diff" in meta_feature_groups:
            _add_pairwise_diff(frame, prediction_columns)

    categorical_columns = ["layout_type"] if "layout_type" in train_meta.columns else []
    feature_columns = [
        column
        for column in train_meta.columns
        if column not in {id_col, group_col, target_col, "layout_id"}
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    for frame in (train_meta, test_meta):
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").astype("float32")

    metadata = {
        "stacking_members": members,
        "stacking_meta_features": sorted(meta_feature_groups),
        "stacking_feature_columns": feature_columns,
    }
    return train_meta, test_meta, feature_columns, categorical_columns, metadata


def run_stacking_cv(config: dict) -> dict:
    prepare_runtime(config)

    data = load_data(config, usecols_map=_get_stacking_usecols_map(config))
    validate_stacking_schema(
        train_df=data["train"],
        test_df=data["test"],
        layout_df=data["layout"],
        sub_df=data["sub"],
    )
    fold_df = make_cv_folds(data["train"], config)
    train_meta, test_meta, feature_columns, categorical_columns, metadata = _build_meta_frames(
        data=data,
        config=config,
    )

    meta_config = deepcopy(config)
    meta_config["model"] = deepcopy(config["stacking"]["meta_model"])
    meta_config.setdefault("training", {})
    meta_config["training"]["target_transform"] = "none"
    meta_config["training"]["sample_weight_mode"] = "none"

    metrics = train_cv(
        train_df=train_meta,
        test_df=test_meta,
        fold_df=fold_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        config=meta_config,
        sample_weights=None,
        extra_metrics=metadata,
    )
    return metrics


def run_stacking(config: dict) -> str:
    meta_config = deepcopy(config)
    run_stacking_cv(meta_config)
    meta_config["model"] = deepcopy(config["stacking"]["meta_model"])
    meta_config.setdefault("training", {})
    meta_config["training"]["target_transform"] = "none"
    meta_config["training"]["sample_weight_mode"] = "none"

    return save_submission(meta_config)
