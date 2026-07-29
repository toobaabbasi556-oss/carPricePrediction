"""CatBoost / LightGBM on log(price_pkr) with native categorical handling (SPEC §5).

Both consume the ``category``-dtype columns emitted by the feature pipeline directly —
no one-hot (SPEC §4). LightGBM auto-detects pandas ``category`` columns; CatBoost needs
the categorical column *names*, and its cat features must be strings, so a tiny cast step
precedes it. Tracked with MLflow in :mod:`pkcar.train` (SPEC §8).
"""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from ..config import load_config
from ..pipeline import build_feature_pipeline, categorical_features


def _params(config: dict | None, key: str, defaults: dict) -> dict:
    cfg = config or load_config()
    return {**defaults, **cfg.get("model", {}).get(key, {})}


def build_lightgbm_model(config: dict | None = None) -> Pipeline:
    """Feature pipeline -> LightGBM (native categorical), on log-price (SPEC §5)."""
    cfg = config or load_config()
    params = _params(cfg, "lgbm", {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": int(cfg["project"]["random_seed"]),
        "n_jobs": -1,
        "verbose": -1,
    })
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline(cfg)),
            ("lgbm", LGBMRegressor(**params)),
        ]
    )


class _CategoryToStr(BaseEstimator, TransformerMixin):
    """Cast the given columns to string — CatBoost cat_features must be str, not category."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> _CategoryToStr:  # noqa: ARG002
        self.columns_ = list(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            out[col] = out[col].astype("string").astype("object")
        return out


def build_catboost_model(config: dict | None = None) -> Pipeline:
    """Feature pipeline -> CatBoost (native categorical), on log-price (SPEC §5)."""
    cfg = config or load_config()
    params = _params(cfg, "catboost", {
        "iterations": 600,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": int(cfg["project"]["random_seed"]),
        "verbose": False,
    })
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline(cfg)),
            ("to_str", _CategoryToStr(categorical_features())),
            ("catboost", CatBoostRegressor(cat_features=categorical_features(), **params)),
        ]
    )
