from __future__ import annotations

import pandas as pd

from .common import get_artifact_paths
from .data_io import get_data_paths


def save_submission(config: dict) -> str:
    id_col = config["columns"]["id_col"]
    target_col = config["columns"]["target_col"]
    artifacts = get_artifact_paths(config)
    data_paths = get_data_paths(config)

    if not artifacts["test_pred_path"].exists():
        raise FileNotFoundError(f"Missing test prediction file: {artifacts['test_pred_path']}")

    sample_submission = pd.read_csv(data_paths["sub"], encoding="utf-8-sig")
    test_prediction = pd.read_csv(artifacts["test_pred_path"], encoding="utf-8-sig")
    if "prediction" not in test_prediction.columns:
        raise ValueError("The test prediction file must contain a `prediction` column.")

    submission = sample_submission[[id_col]].merge(
        test_prediction[[id_col, "prediction"]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )
    if submission["prediction"].isna().any():
        raise ValueError("Submission contains missing predictions.")

    submission = submission.rename(columns={"prediction": target_col})
    submission.to_csv(artifacts["submission_path"], index=False, encoding="utf-8-sig")
    print(f"[submission] saved={artifacts['submission_path']}")
    return str(artifacts["submission_path"])
