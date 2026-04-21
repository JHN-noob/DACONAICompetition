from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_float = pd.to_numeric(numerator, errors="coerce").astype("float32")
    denominator_float = pd.to_numeric(denominator, errors="coerce").astype("float32")
    denominator_safe = denominator_float.mask(denominator_float == 0, np.nan)
    return (numerator_float / denominator_safe).astype("float32")


def _to_float32(frame: pd.DataFrame, columns: list[str]) -> None:
    if columns:
        frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce").astype("float32")


def _concat_feature_block(frame: pd.DataFrame, features: dict[str, pd.Series]) -> pd.DataFrame:
    if not features:
        return frame
    feature_frame = pd.DataFrame(features, index=frame.index)
    feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce").astype("float32")
    return pd.concat([frame, feature_frame], axis=1, copy=False)


def _fill_categorical_missing(frame: pd.DataFrame, column: str, missing_label: str = "__MISSING__") -> None:
    if is_categorical_dtype(frame[column]):
        if missing_label not in frame[column].cat.categories:
            frame[column] = frame[column].cat.add_categories([missing_label])
        frame[column] = frame[column].fillna(missing_label)
    else:
        frame[column] = frame[column].fillna(missing_label).astype("category")


def _add_row_stats(frame: pd.DataFrame, raw_feature_columns: list[str]) -> pd.DataFrame:
    values = frame[raw_feature_columns]
    row_missing_count = values.isna().sum(axis=1).astype("float32")
    features = {
        "row_mean": values.mean(axis=1).astype("float32"),
        "row_std": values.std(axis=1).astype("float32"),
        "row_min": values.min(axis=1).astype("float32"),
        "row_max": values.max(axis=1).astype("float32"),
        "row_median": values.median(axis=1).astype("float32"),
        "row_q25": values.quantile(0.25, axis=1).astype("float32"),
        "row_q75": values.quantile(0.75, axis=1).astype("float32"),
        "row_missing_count": row_missing_count,
        "row_missing_ratio": (row_missing_count / float(len(raw_feature_columns))).astype("float32"),
    }
    return _concat_feature_block(frame, features)


def _build_scenario_stats(frame: pd.DataFrame, raw_feature_columns: list[str], group_col: str) -> pd.DataFrame:
    aggregations = ["mean", "std", "min", "max"]
    grouped = frame.groupby(group_col)[raw_feature_columns].agg(aggregations)
    grouped.columns = [f"{column}_scenario_{agg}" for column, agg in grouped.columns]
    grouped = grouped.reset_index()
    stat_columns = [column for column in grouped.columns if column != group_col]
    grouped[stat_columns] = grouped[stat_columns].astype("float32")
    return grouped


def _add_scenario_relative_features(frame: pd.DataFrame, raw_feature_columns: list[str]) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for column in raw_feature_columns:
        mean_column = f"{column}_scenario_mean"
        std_column = f"{column}_scenario_std"
        min_column = f"{column}_scenario_min"
        max_column = f"{column}_scenario_max"

        if mean_column in frame.columns:
            features[f"{column}_scenario_delta"] = (frame[column] - frame[mean_column]).astype("float32")

        if mean_column in frame.columns and std_column in frame.columns:
            std_safe = pd.to_numeric(frame[std_column], errors="coerce").astype("float32").mask(frame[std_column] == 0, np.nan)
            zscore_values = (frame[column] - frame[mean_column]) / std_safe
            features[f"{column}_scenario_zscore"] = zscore_values.astype("float32").fillna(0.0)

        if min_column in frame.columns and max_column in frame.columns:
            range_safe = (frame[max_column] - frame[min_column]).astype("float32").mask(frame[max_column] == frame[min_column], np.nan)
            pos_values = (frame[column] - frame[min_column]) / range_safe
            features[f"{column}_scenario_minmax_pos"] = pos_values.astype("float32").fillna(0.0)

    return _concat_feature_block(frame, features)


def _add_layout_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("order_inflow_15m", "pack_station_count", "order_inflow_per_pack_station"),
        ("unique_sku_15m", "pack_station_count", "unique_sku_per_pack_station"),
        ("pack_utilization", "pack_station_count", "pack_utilization_per_pack_station"),
        ("robot_idle", "robot_total", "robot_idle_ratio_layout"),
        ("robot_charging", "charger_count", "robot_charging_per_charger"),
        ("charge_queue_length", "charger_count", "charge_queue_per_charger"),
        ("congestion_score", "intersection_count", "congestion_per_intersection"),
        ("blocked_path_15m", "intersection_count", "blocked_path_per_intersection"),
    ]
    features: dict[str, pd.Series] = {}
    for numerator, denominator, output_column in specs:
        if numerator in frame.columns and denominator in frame.columns:
            features[output_column] = _safe_ratio(frame[numerator], frame[denominator])

    if {"pack_utilization", "layout_compactness"}.issubset(frame.columns):
        features["pack_utilization_x_layout_compactness"] = (
            frame["pack_utilization"] * frame["layout_compactness"]
        ).astype("float32")
    if {"congestion_score", "zone_dispersion"}.issubset(frame.columns):
        features["congestion_x_zone_dispersion"] = (
            frame["congestion_score"] * frame["zone_dispersion"]
        ).astype("float32")

    return _concat_feature_block(frame, features)


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    id_col = config["columns"]["id_col"]
    group_col = config["columns"]["group_col"]
    target_col = config["columns"]["target_col"]

    train = train_df.copy()
    test = test_df.copy()
    layout = layout_df.copy()
    feature_config = config.get("features", {})
    use_layout = bool(feature_config.get("use_layout", True))
    use_layout_id = bool(feature_config.get("use_layout_id", False))
    use_layout_type = bool(feature_config.get("use_layout_type", True))
    use_layout_interaction_features = bool(feature_config.get("use_layout_interaction_features", True))

    raw_feature_columns = [
        column
        for column in train.columns
        if column not in {id_col, "layout_id", group_col, target_col}
    ]

    _to_float32(train, raw_feature_columns)
    _to_float32(test, raw_feature_columns)

    if use_layout:
        layout_numeric_columns = [
            column for column in layout.columns if column not in {"layout_id", "layout_type"}
        ]
        _to_float32(layout, layout_numeric_columns)
        train = train.merge(layout, on="layout_id", how="left", validate="many_to_one")
        test = test.merge(layout, on="layout_id", how="left", validate="many_to_one")
        if use_layout_interaction_features:
            train = _add_layout_interaction_features(train)
            test = _add_layout_interaction_features(test)

    if feature_config.get("use_row_stats", True):
        train = _add_row_stats(train, raw_feature_columns)
        test = _add_row_stats(test, raw_feature_columns)

    if feature_config.get("use_scenario_stats", True):
        train_stats = _build_scenario_stats(train, raw_feature_columns, group_col)
        test_stats = _build_scenario_stats(test, raw_feature_columns, group_col)
        train = train.merge(train_stats, on=group_col, how="left", validate="many_to_one")
        test = test.merge(test_stats, on=group_col, how="left", validate="many_to_one")
        if bool(feature_config.get("use_scenario_relative_features", False)):
            train = _add_scenario_relative_features(train, raw_feature_columns)
            test = _add_scenario_relative_features(test, raw_feature_columns)

    drop_columns = []
    if not use_layout_id and "layout_id" in train.columns:
        drop_columns.append("layout_id")
    if not use_layout_type and "layout_type" in train.columns:
        drop_columns.append("layout_type")
    if drop_columns:
        train = train.drop(columns=drop_columns)
        test = test.drop(columns=drop_columns)

    categorical_columns = [column for column in ["layout_id", "layout_type"] if column in train.columns]
    for frame in (train, test):
        for column in categorical_columns:
            _fill_categorical_missing(frame, column)

    feature_columns = [
        column for column in train.columns if column not in {id_col, group_col, target_col}
    ]
    numeric_feature_columns = [column for column in feature_columns if column not in categorical_columns]
    _to_float32(train, numeric_feature_columns)
    _to_float32(test, numeric_feature_columns)

    return train, test, feature_columns, categorical_columns
