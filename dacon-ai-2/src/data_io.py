from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .common import resolve_path


def get_data_paths(config: dict) -> dict[str, Path]:
    data_dir = resolve_path(config["paths"]["data_dir"])
    paths = {
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "layout": data_dir / "layout_info.csv",
        "sub": data_dir / "sample_submission.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")
    return paths


def load_data(
    config: dict,
    usecols_map: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    paths = get_data_paths(config)
    frames: dict[str, pd.DataFrame] = {}
    for key, path in paths.items():
        read_kwargs = {"encoding": "utf-8-sig"}
        columns = None if usecols_map is None else usecols_map.get(key)
        if columns is not None:
            read_kwargs["usecols"] = list(dict.fromkeys(columns))
        frames[key] = pd.read_csv(path, **read_kwargs)
    return frames
