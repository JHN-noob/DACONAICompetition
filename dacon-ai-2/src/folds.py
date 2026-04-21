from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _build_strata(meta_df: pd.DataFrame, target_column: str, n_bins: int) -> pd.Series:
    unique_target_count = int(meta_df[target_column].nunique())
    effective_bins = min(n_bins, unique_target_count)
    if effective_bins < 2:
        return pd.Series(0, index=meta_df.index, dtype="int64")

    strata = pd.qcut(
        meta_df[target_column],
        q=effective_bins,
        labels=False,
        duplicates="drop",
    )
    strata = pd.Series(strata, index=meta_df.index)
    if strata.nunique() < 2:
        return pd.Series(0, index=meta_df.index, dtype="int64")
    return strata.astype(int)


def make_cv_folds(train_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    id_col = config["columns"]["id_col"]
    group_col = config["columns"]["group_col"]
    target_col = config["columns"]["target_col"]
    n_splits = int(config["cv"]["n_splits"])
    n_bins = int(config["cv"].get("n_bins", 10))
    shuffle = bool(config["cv"].get("shuffle", True))
    random_state = int(config["cv"].get("random_state", 42))
    strategy = str(config["cv"].get("strategy", "scenario_stratified")).lower()

    if strategy == "scenario_stratified":
        split_col = group_col
    elif strategy == "layout_holdout":
        split_col = "layout_id"
    else:
        raise ValueError(f"Unsupported cv strategy: {strategy}")

    split_meta = (
        train_df.groupby(split_col, as_index=False)[target_col]
        .mean()
        .rename(columns={target_col: "target_mean"})
    )
    split_meta["strata"] = _build_strata(split_meta, "target_mean", n_bins)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    split_meta["fold"] = -1

    for fold, (_, valid_idx) in enumerate(
        splitter.split(split_meta[split_col], split_meta["strata"])
    ):
        split_meta.loc[split_meta.index[valid_idx], "fold"] = fold

    if (split_meta["fold"] < 0).any():
        raise ValueError("Failed to assign folds for all scenarios.")

    fold_df = train_df[[id_col, split_col]].merge(
        split_meta[[split_col, "fold"]],
        on=split_col,
        how="left",
        validate="many_to_one",
    )
    if fold_df["fold"].isna().any():
        raise ValueError("Missing fold assignments for some training rows.")

    fold_df["fold"] = fold_df["fold"].astype(int)
    return fold_df[[id_col, "fold"]].sort_values(id_col).reset_index(drop=True)
