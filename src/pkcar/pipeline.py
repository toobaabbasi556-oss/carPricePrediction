"""The sklearn feature Pipeline, fit INSIDE CV folds (SPEC §5).

Never fit any transform on the full dataset — that is leakage. This module wires the
SPEC §4 feature layer into one ``sklearn.Pipeline`` whose stateful steps (rare-category
sets, numeric medians) learn only from the training rows they are fit on. The estimator
is attached in the model layer (Phase 5); the output here is a clean feature frame with
categoricals as ``category`` dtype for native-categorical boosters (SPEC §4).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .config import load_config
from .features.derived import (
    DerivedFeatures,
    NumericMedianImputer,
    RareCategoryGrouper,
)

# Post-feature-engineering model schema (SPEC §4). Excludes ids, target, and the
# leakage column (canonical_price_pkr) by construction — they are never selected.
MODEL_CATEGORICAL: tuple[str, ...] = (
    "make",
    "model",
    "city",
    "registration_city",
    "body_type",
    "color_norm",
    "fuel",
    "transmission",
    "engine_cc_bucket",
    "assembled_vs_imported",
)

MODEL_NUMERIC: tuple[str, ...] = (
    "year",
    "age",
    "mileage_km",
    "engine_cc",
    "assembly_confidence",
    "seller_is_dealer",
)

MODEL_FEATURES: tuple[str, ...] = MODEL_CATEGORICAL + MODEL_NUMERIC


def categorical_features() -> list[str]:
    """Native-categorical model features (SPEC §4)."""
    return list(MODEL_CATEGORICAL)


def numeric_features() -> list[str]:
    """Numeric model features (SPEC §4)."""
    return list(MODEL_NUMERIC)


def model_features() -> list[str]:
    """All model features, categorical first (stable column order)."""
    return list(MODEL_FEATURES)


class SelectColumns(BaseEstimator, TransformerMixin):
    """Select and order the final model columns; drops ids/target/leakage by omission."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> SelectColumns:  # noqa: ARG002
        self.columns_ = list(self.columns)  # fitted marker for sklearn check_is_fitted
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.columns].copy()


def build_feature_pipeline(
    config: dict | None = None,
    rare_min_count: int | None = None,
) -> Pipeline:
    """Build the leakage-safe feature pipeline (SPEC §4, §5).

    Order: derived features -> rare/unknown category grouping -> numeric median
    imputation -> final column selection. Every stateful step is fit on the training
    fold only when this pipeline is fit inside CV.
    """
    cfg = config or load_config()
    current_year = int(cfg["project"]["current_year"])
    min_count = rare_min_count if rare_min_count is not None else int(
        cfg.get("features", {}).get("rare_min_count", 10)
    )
    return Pipeline(
        steps=[
            ("derived", DerivedFeatures(current_year=current_year)),
            ("rare", RareCategoryGrouper(columns=list(MODEL_CATEGORICAL), min_count=min_count)),
            ("impute", NumericMedianImputer(columns=list(MODEL_NUMERIC))),
            ("select", SelectColumns(columns=list(MODEL_FEATURES))),
        ]
    )
