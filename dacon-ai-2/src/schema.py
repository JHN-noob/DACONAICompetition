from __future__ import annotations

import pandas as pd

ID_COL = "ID"
GROUP_COL = "scenario_id"
TARGET_COL = "avg_delay_minutes_next_30m"
LAYOUT_ID_COL = "layout_id"


def _assert_columns_exist(df: pd.DataFrame, required_columns: list[str], name: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _assert_unique(df: pd.DataFrame, column: str, name: str) -> None:
    if not df[column].is_unique:
        raise ValueError(f"{name} column `{column}` must be unique.")


def _assert_group_size(df: pd.DataFrame, group_col: str, expected_size: int, name: str) -> None:
    group_sizes = df.groupby(group_col).size()
    if not group_sizes.eq(expected_size).all():
        values = group_sizes.value_counts().to_dict()
        raise ValueError(f"{name} group size mismatch for `{group_col}`: {values}")


def validate_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> None:
    train_required = [ID_COL, LAYOUT_ID_COL, GROUP_COL, TARGET_COL]
    test_required = [ID_COL, LAYOUT_ID_COL, GROUP_COL]
    layout_required = [LAYOUT_ID_COL, "layout_type"]
    sub_required = [ID_COL, TARGET_COL]

    _assert_columns_exist(train_df, train_required, "train")
    _assert_columns_exist(test_df, test_required, "test")
    _assert_columns_exist(layout_df, layout_required, "layout")
    _assert_columns_exist(sub_df, sub_required, "sample_submission")

    if len(train_df.columns) != 94:
        raise ValueError(f"train column count mismatch: {len(train_df.columns)} != 94")
    if len(test_df.columns) != 93:
        raise ValueError(f"test column count mismatch: {len(test_df.columns)} != 93")
    if len(layout_df.columns) != 15:
        raise ValueError(f"layout column count mismatch: {len(layout_df.columns)} != 15")
    if len(sub_df.columns) != 2:
        raise ValueError(f"sample_submission column count mismatch: {len(sub_df.columns)} != 2")

    _assert_unique(train_df, ID_COL, "train")
    _assert_unique(test_df, ID_COL, "test")
    _assert_unique(layout_df, LAYOUT_ID_COL, "layout")
    _assert_unique(sub_df, ID_COL, "sample_submission")

    if TARGET_COL in test_df.columns:
        raise ValueError("test must not contain the target column.")

    train_feature_columns = sorted(column for column in train_df.columns if column != TARGET_COL)
    test_feature_columns = sorted(test_df.columns.tolist())
    if train_feature_columns != test_feature_columns:
        raise ValueError("train/test feature columns do not match after excluding the target.")

    if sorted(sub_df[ID_COL].tolist()) != sorted(test_df[ID_COL].tolist()):
        raise ValueError("sample_submission IDs do not match test IDs.")

    _assert_group_size(train_df, GROUP_COL, expected_size=25, name="train")
    _assert_group_size(test_df, GROUP_COL, expected_size=25, name="test")

    missing_layout_ids = sorted(
        set(pd.concat([train_df[LAYOUT_ID_COL], test_df[LAYOUT_ID_COL]])).difference(layout_df[LAYOUT_ID_COL])
    )
    if missing_layout_ids:
        raise ValueError(f"Missing layout_id values in layout_info.csv: {missing_layout_ids[:10]}")


def validate_stacking_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    layout_df: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> None:
    _assert_columns_exist(train_df, [ID_COL, LAYOUT_ID_COL, GROUP_COL, TARGET_COL], "train")
    _assert_columns_exist(test_df, [ID_COL, LAYOUT_ID_COL, GROUP_COL], "test")
    _assert_columns_exist(layout_df, [LAYOUT_ID_COL], "layout")
    _assert_columns_exist(sub_df, [ID_COL, TARGET_COL], "sample_submission")

    _assert_unique(train_df, ID_COL, "train")
    _assert_unique(test_df, ID_COL, "test")
    _assert_unique(layout_df, LAYOUT_ID_COL, "layout")
    _assert_unique(sub_df, ID_COL, "sample_submission")

    if TARGET_COL in test_df.columns:
        raise ValueError("test must not contain the target column.")

    if sorted(sub_df[ID_COL].tolist()) != sorted(test_df[ID_COL].tolist()):
        raise ValueError("sample_submission IDs do not match test IDs.")

    _assert_group_size(train_df, GROUP_COL, expected_size=25, name="train")
    _assert_group_size(test_df, GROUP_COL, expected_size=25, name="test")

    missing_layout_ids = sorted(
        set(pd.concat([train_df[LAYOUT_ID_COL], test_df[LAYOUT_ID_COL]])).difference(layout_df[LAYOUT_ID_COL])
    )
    if missing_layout_ids:
        raise ValueError(f"Missing layout_id values in layout_info.csv: {missing_layout_ids[:10]}")
