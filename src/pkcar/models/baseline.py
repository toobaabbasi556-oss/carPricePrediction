"""Baseline Linear Regression on log(price_pkr) (SPEC §5). A genuine baseline.

The feature pipeline emits ``category``-dtype categoricals for native-categorical
boosters; a linear model cannot consume those, so this baseline adds a one-hot +
standard-scale step. That encoder is fit **inside** the pipeline (hence inside CV
folds, SPEC §5). ``handle_unknown="ignore"`` is a second safety net on top of the
rare/unknown grouping already done upstream.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..pipeline import build_feature_pipeline, categorical_features, numeric_features


def build_baseline_model(config: dict | None = None) -> Pipeline:
    """Feature pipeline -> one-hot + scale -> LinearRegression, all fit inside folds."""
    encode = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True),
             categorical_features()),
            ("num", StandardScaler(), numeric_features()),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("features", build_feature_pipeline(config)),
            ("encode", encode),
            ("lr", LinearRegression()),
        ]
    )
