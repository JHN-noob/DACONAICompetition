from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _encode_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df[feature_columns].copy()
    test = test_df[feature_columns].copy()
    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_categorical_dtype(train_df[column])
    ]

    for column in categorical_columns:
        combined = pd.concat([train[column].astype(str), test[column].astype(str)], ignore_index=True)
        categories = pd.Index(combined.unique())
        mapping = {value: idx for idx, value in enumerate(categories)}
        train[column] = train[column].astype(str).map(mapping).astype("int32")
        test[column] = test[column].astype(str).map(mapping).astype("int32")

    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    for frame in (train, test):
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
        frame[categorical_columns] = frame[categorical_columns].astype("float32")

    return train, test


def build_domain_weights(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
) -> pd.Series:
    id_col = config["columns"]["id_col"]
    seed = int(config.get("seed", 42))
    domain_config = config.get("domain_adaptation", {})
    n_splits = int(domain_config.get("n_splits", 5))
    weight_clip_low = float(domain_config.get("weight_clip_low", 0.5))
    weight_clip_high = float(domain_config.get("weight_clip_high", 1.5))
    classifier_params = domain_config.get("classifier_params", {})

    encoded_train, encoded_test = _encode_feature_frames(train_df, test_df, feature_columns)
    domain_frame = pd.concat([encoded_train, encoded_test], axis=0, ignore_index=True)
    domain_target = np.concatenate(
        [
            np.zeros(len(encoded_train), dtype=np.int8),
            np.ones(len(encoded_test), dtype=np.int8),
        ]
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    oof_probabilities = np.zeros(len(domain_frame), dtype=np.float64)
    fold_aucs: list[float] = []

    default_params = {
        "max_depth": 6,
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 80,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
        "random_state": seed,
    }
    default_params.update(classifier_params)

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(domain_frame, domain_target)):
        model = HistGradientBoostingClassifier(**default_params)
        model.fit(domain_frame.iloc[fit_idx], domain_target[fit_idx])
        valid_probability = model.predict_proba(domain_frame.iloc[valid_idx])[:, 1]
        oof_probabilities[valid_idx] = valid_probability
        fold_auc = float(roc_auc_score(domain_target[valid_idx], valid_probability))
        fold_aucs.append(fold_auc)
        print(f"[domain] fold={fold} valid_auc={fold_auc:.6f}")

    train_probabilities = oof_probabilities[: len(encoded_train)]
    weights = np.clip(0.5 + train_probabilities, a_min=weight_clip_low, a_max=weight_clip_high)
    weights = weights / weights.mean()

    weight_series = pd.Series(
        weights.astype("float32"),
        index=train_df[id_col].copy(),
        name="sample_weight",
    )
    weight_series.attrs["domain_metrics"] = {
        "fold_auc": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "train_probability_mean": float(train_probabilities.mean()),
        "train_probability_std": float(train_probabilities.std(ddof=0)),
        "weight_clip_low": weight_clip_low,
        "weight_clip_high": weight_clip_high,
    }
    return weight_series
