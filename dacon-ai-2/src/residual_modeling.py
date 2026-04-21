from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .common import prepare_runtime, resolve_path
from .data_io import load_data
from .features import build_features
from .folds import make_cv_folds
from .schema import validate_schema
from .train import train_cv


def load_run_predictions(config: dict, run_name: str, split: str) -> pd.DataFrame:
    output_dir = resolve_path(config["paths"]["output_dir"])
    path = output_dir / "oof" / f"{run_name}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing residual artifact: {path}. "
            "Run the required upstream config first to create OOF/test predictions."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def merge_prediction_feature(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    run_name: str,
    feature_name: str,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pred = load_run_predictions(config, run_name, "oof")
    test_pred = load_run_predictions(config, run_name, "test")
    train_column = train_pred[[id_col, "prediction"]].rename(columns={"prediction": feature_name})
    test_column = test_pred[[id_col, "prediction"]].rename(columns={"prediction": feature_name})

    train_df = train_df.merge(train_column, on=id_col, how="left", validate="one_to_one")
    test_df = test_df.merge(test_column, on=id_col, how="left", validate="one_to_one")
    if train_df[feature_name].isna().any() or test_df[feature_name].isna().any():
        raise ValueError(f"Residual prediction feature `{feature_name}` is missing for some rows.")

    train_df[feature_name] = pd.to_numeric(train_df[feature_name], errors="coerce").astype("float32")
    test_df[feature_name] = pd.to_numeric(test_df[feature_name], errors="coerce").astype("float32")
    return train_df, test_df


def run_residual_cv(config: dict) -> dict:
    prepare_runtime(config)

    data = load_data(config)
    validate_schema(
        train_df=data["train"],
        test_df=data["test"],
        layout_df=data["layout"],
        sub_df=data["sub"],
    )
    fold_df = make_cv_folds(data["train"], config)
    train_df, test_df, feature_columns, categorical_columns = build_features(
        train_df=data["train"],
        test_df=data["test"],
        layout_df=data["layout"],
        config=config,
    )

    id_col = config["columns"]["id_col"]
    target_col = config["columns"]["target_col"]
    residual_config = config["residual"]
    base_run_name = str(residual_config["base_run_name"])
    base_feature_name = str(residual_config.get("base_prediction_feature_name", "base_prediction"))
    use_base_prediction_feature = bool(residual_config.get("use_base_prediction_feature", True))

    train_df, test_df = merge_prediction_feature(
        train_df=train_df,
        test_df=test_df,
        config=config,
        run_name=base_run_name,
        feature_name=base_feature_name,
        id_col=id_col,
    )

    if use_base_prediction_feature and base_feature_name not in feature_columns:
        feature_columns = feature_columns + [base_feature_name]

    extra_prediction_features = residual_config.get("extra_prediction_features", [])
    extra_feature_names: list[str] = []
    for prediction_feature in extra_prediction_features:
        run_name = str(prediction_feature["run_name"])
        feature_name = str(prediction_feature["feature_name"])
        train_df, test_df = merge_prediction_feature(
            train_df=train_df,
            test_df=test_df,
            config=config,
            run_name=run_name,
            feature_name=feature_name,
            id_col=id_col,
        )
        extra_feature_names.append(feature_name)
        if feature_name not in feature_columns:
            feature_columns = feature_columns + [feature_name]

    if bool(residual_config.get("generate_prediction_diffs", False)):
        prediction_feature_names = [base_feature_name, *extra_feature_names]
        for left_idx, left_feature_name in enumerate(prediction_feature_names):
            for right_feature_name in prediction_feature_names[left_idx + 1 :]:
                diff_feature_name = f"diff__{left_feature_name}__{right_feature_name}"
                train_df[diff_feature_name] = (
                    train_df[left_feature_name] - train_df[right_feature_name]
                ).astype("float32")
                test_df[diff_feature_name] = (
                    test_df[left_feature_name] - test_df[right_feature_name]
                ).astype("float32")
                if diff_feature_name not in feature_columns:
                    feature_columns = feature_columns + [diff_feature_name]

    feature_subset = residual_config.get("feature_subset")
    if feature_subset:
        missing_features = [feature_name for feature_name in feature_subset if feature_name not in train_df.columns]
        if missing_features:
            raise ValueError(
                f"Some residual.feature_subset columns are missing from the feature frame: {missing_features}"
            )
        feature_columns = list(feature_subset)
        categorical_columns = [column for column in categorical_columns if column in feature_columns]

    fit_target_values = (train_df[target_col] - train_df[base_feature_name]).astype("float32")
    prediction_offset_train = train_df.set_index(id_col)[base_feature_name].astype("float32")
    prediction_offset_test = test_df.set_index(id_col)[base_feature_name].astype("float32")

    residual_metadata = {
        "residual": {
            "base_run_name": base_run_name,
            "base_prediction_feature_name": base_feature_name,
            "use_base_prediction_feature": use_base_prediction_feature,
            "extra_prediction_features": extra_prediction_features,
            "generate_prediction_diffs": bool(residual_config.get("generate_prediction_diffs", False)),
            "feature_subset": feature_subset,
        }
    }

    residual_train_config = deepcopy(config)
    residual_train_config.setdefault("training", {})
    residual_train_config["training"]["target_transform"] = "none"
    residual_train_config["training"]["sample_weight_mode"] = "none"

    return train_cv(
        train_df=train_df,
        test_df=test_df,
        fold_df=fold_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        config=residual_train_config,
        sample_weights=None,
        extra_metrics=residual_metadata,
        fit_target_values=fit_target_values.set_axis(train_df[id_col]),
        prediction_offset_train=prediction_offset_train,
        prediction_offset_test=prediction_offset_test,
    )
