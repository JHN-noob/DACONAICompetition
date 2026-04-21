from __future__ import annotations

from copy import deepcopy


def get_model_type(config: dict) -> str:
    return str(config["model"]["type"]).lower()


def get_model_params(config: dict) -> dict:
    return deepcopy(config["model"].get("params", {}))


def build_model(config: dict, params_override: dict | None = None):
    model_type = get_model_type(config)
    if model_type != "lightgbm":
        raise ValueError(f"Only LightGBM is supported in the final pipeline: {model_type}")

    from lightgbm import LGBMRegressor

    params = deepcopy(params_override) if params_override is not None else get_model_params(config)
    params.pop("early_stopping_rounds", None)
    params.pop("verbose_eval", None)
    return LGBMRegressor(**params)
