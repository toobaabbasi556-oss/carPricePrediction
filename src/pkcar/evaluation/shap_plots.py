"""SHAP summary + dependence plots to explain price drivers (SPEC §6).

SHAP values come from CatBoost's native ``get_feature_importance(type="ShapValues")`` on
the *transformed* feature frame (the columns the booster actually sees, after the feature
pipeline). Values are on the log-price scale — the model's training target — so they read
as additive contributions to log(price).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import Pool
from sklearn.pipeline import Pipeline

from ..pipeline import categorical_features


def transform_for_catboost(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run X through every pipeline step except the final CatBoost estimator."""
    return model[:-1].transform(X)


def compute_shap_values(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, float]:
    """Return ``(shap_values [n, f], transformed_features, expected_value)`` (log scale)."""
    cb = model.named_steps["catboost"]
    xt = transform_for_catboost(model, X)
    pool = Pool(xt, cat_features=categorical_features())
    raw = cb.get_feature_importance(pool, type="ShapValues")  # (n, f+1); last col = base value
    shap_values = raw[:, :-1]
    expected_value = float(raw[0, -1])
    return shap_values, xt, expected_value


def save_shap_summary(shap_values: np.ndarray, xt: pd.DataFrame, path: str | Path) -> Path:
    """SHAP beeswarm summary of global price drivers (SPEC §6)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, xt, show=False, max_display=15)
    plt.title("SHAP summary — drivers of log(price)")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


def save_shap_dependence(
    shap_values: np.ndarray, xt: pd.DataFrame, feature: str, path: str | Path
) -> Path:
    """SHAP dependence plot for one feature (SPEC §6)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.dependence_plot(feature, shap_values, xt, show=False, interaction_index=None)
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    return path


def mean_abs_shap(shap_values: np.ndarray, xt: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature — the global importance ranking (SPEC §6)."""
    return pd.Series(np.abs(shap_values).mean(axis=0), index=xt.columns).sort_values(
        ascending=False
    )
