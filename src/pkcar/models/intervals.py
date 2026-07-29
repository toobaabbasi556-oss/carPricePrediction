"""Prediction intervals via split-conformal prediction (SPEC §6).

Why conformal (not MAPIE, not quantile here): it is model-agnostic and gives a
finite-sample marginal coverage guarantee without retraining or cloning the estimator —
CatBoost (our best model) is not sklearn-clone-safe, which rules MAPIE out cleanly.

Mechanics: split the training data into a *proper* fit set and a *calibration* set —
**grouped** so no vehicle spans both (else calibration residuals leak, SPEC §5). Fit the
base model on the proper set (log-price), take the conformal quantile of absolute
calibration residuals on the log scale, and form a symmetric band in log-space. Because
the band is additive in log, it is **multiplicative in PKR** (wider for expensive cars),
which suits right-skewed prices. Report empirical coverage + mean width (SPEC §7).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from ..config import load_config
from ..data.loaders import inverse_log_target, log_target

ModelBuilder = Callable[[dict | None], Pipeline]


def conformal_quantile_level(n_calib: int, alpha: float) -> float:
    """Finite-sample conformal level ceil((n+1)(1-alpha))/n, capped at 1 (SPEC §6)."""
    level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    return float(min(level, 1.0))


class ConformalIntervalModel:
    """Split-conformal prediction intervals around a log-price regressor (SPEC §6, §7)."""

    def __init__(
        self,
        build_model: ModelBuilder,
        alpha: float | None = None,
        calib_size: float = 0.25,
        config: dict | None = None,
    ) -> None:
        self.build_model = build_model
        self.config = config or load_config()
        self.alpha = alpha if alpha is not None else float(self.config["model"]["interval_alpha"])
        self.calib_size = calib_size

    def fit(
        self, X: pd.DataFrame, y_price: np.ndarray, groups: pd.Series
    ) -> ConformalIntervalModel:
        seed = int(self.config["project"]["random_seed"])
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.calib_size, random_state=seed)
        proper_idx, calib_idx = next(splitter.split(X, groups=groups))

        y_log = log_target(y_price)
        self.model_ = self.build_model(self.config)
        self.model_.fit(X.iloc[proper_idx], y_log[proper_idx])

        calib_pred = self.model_.predict(X.iloc[calib_idx])
        residuals = np.abs(y_log[calib_idx] - calib_pred)
        level = conformal_quantile_level(len(calib_idx), self.alpha)
        self.q_ = float(np.quantile(residuals, level, method="higher"))
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(point_pkr, lower_pkr, upper_pkr)``. Band is multiplicative in PKR."""
        pred_log = self.model_.predict(X)
        point = inverse_log_target(pred_log)
        lower = inverse_log_target(pred_log - self.q_)
        upper = inverse_log_target(pred_log + self.q_)
        return point, lower, upper
