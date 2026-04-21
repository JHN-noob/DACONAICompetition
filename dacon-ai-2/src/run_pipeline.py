from __future__ import annotations

from .common import ensure_output_dirs, prepare_runtime
from .data_io import load_data
from .domain_adaptation import build_domain_weights
from .features import build_features
from .folds import make_cv_folds
from .inference import save_submission
from .residual_modeling import merge_prediction_feature, run_residual_cv
from .schema import validate_schema
from .train import train_cv


def run_cv(config: dict) -> dict:
    if bool(config.get("residual", {}).get("enabled", False)):
        return run_residual_cv(config)

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

    prediction_feature_config = config.get("prediction_features", {})
    prediction_feature_metadata = None
    if bool(prediction_feature_config.get("enabled", False)):
        id_col = config["columns"]["id_col"]
        merged_prediction_features: list[str] = []
        for prediction_feature in prediction_feature_config.get("features", []):
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
            merged_prediction_features.append(feature_name)
            if feature_name not in feature_columns:
                feature_columns = feature_columns + [feature_name]

        if bool(prediction_feature_config.get("generate_prediction_diffs", False)):
            for left_idx, left_feature_name in enumerate(merged_prediction_features):
                for right_feature_name in merged_prediction_features[left_idx + 1 :]:
                    diff_feature_name = f"diff__{left_feature_name}__{right_feature_name}"
                    train_df[diff_feature_name] = (
                        train_df[left_feature_name] - train_df[right_feature_name]
                    ).astype("float32")
                    test_df[diff_feature_name] = (
                        test_df[left_feature_name] - test_df[right_feature_name]
                    ).astype("float32")
                    if diff_feature_name not in feature_columns:
                        feature_columns = feature_columns + [diff_feature_name]

        feature_subset = prediction_feature_config.get("feature_subset")
        if feature_subset:
            missing_features = [feature_name for feature_name in feature_subset if feature_name not in train_df.columns]
            if missing_features:
                raise ValueError(
                    "Some prediction_features.feature_subset columns are missing from the feature frame: "
                    f"{missing_features}"
                )
            feature_columns = list(feature_subset)
            categorical_columns = [column for column in categorical_columns if column in feature_columns]

        prediction_feature_metadata = {
            "prediction_features": {
                "features": prediction_feature_config.get("features", []),
                "generate_prediction_diffs": bool(prediction_feature_config.get("generate_prediction_diffs", False)),
                "feature_subset": prediction_feature_config.get("feature_subset"),
            }
        }

    sample_weight_mode = str(config.get("training", {}).get("sample_weight_mode", "none")).lower()
    sample_weights = None
    if sample_weight_mode == "adversarial_test_likelihood":
        sample_weights = build_domain_weights(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            config=config,
        )
    elif sample_weight_mode not in {"", "none"}:
        raise ValueError(f"Unsupported sample_weight_mode: {sample_weight_mode}")

    extra_metrics = prediction_feature_metadata if prediction_feature_metadata else None
    return train_cv(
        train_df=train_df,
        test_df=test_df,
        fold_df=fold_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        config=config,
        sample_weights=sample_weights,
        extra_metrics=extra_metrics,
    )


def run_submission(config: dict) -> str:
    ensure_output_dirs(config)
    return save_submission(config)
