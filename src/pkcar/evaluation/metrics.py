"""Point metrics in raw PKR — MAE, RMSE, MAPE — plus interval coverage + mean
width (SPEC §7). Predictions trained on log-price are exponentiated back to PKR.
"""

from __future__ import annotations

import numpy as np

from ..data.loaders import inverse_log_target


def regression_metrics_pkr(
    y_true_pkr: np.ndarray, y_pred_pkr: np.ndarray
) -> dict[str, float]:
    """MAE, RMSE, MAPE in raw PKR (SPEC §7). MAPE as a fraction (0.18 = 18%)."""
    yt = np.asarray(y_true_pkr, dtype=float)
    yp = np.asarray(y_pred_pkr, dtype=float)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / yt))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def metrics_from_log_predictions(
    y_true_log: np.ndarray, y_pred_log: np.ndarray
) -> dict[str, float]:
    """Invert log-price predictions to PKR, then score (SPEC §5: metrics in raw PKR)."""
    return regression_metrics_pkr(
        inverse_log_target(y_true_log), inverse_log_target(y_pred_log)
    )


def interval_metrics_pkr(
    y_true_pkr: np.ndarray, lower_pkr: np.ndarray, upper_pkr: np.ndarray
) -> dict[str, float]:
    """Empirical coverage + mean interval width in PKR (SPEC §6, §7)."""
    yt = np.asarray(y_true_pkr, dtype=float)
    lo = np.asarray(lower_pkr, dtype=float)
    hi = np.asarray(upper_pkr, dtype=float)
    coverage = float(np.mean((yt >= lo) & (yt <= hi)))
    mean_width = float(np.mean(hi - lo))
    return {"coverage": coverage, "mean_width": mean_width}


def format_pkr(value: float) -> str:
    """Human-readable PKR (lacs/crore) for reports."""
    if value >= 1e7:
        return f"PKR {value / 1e7:.2f} crore"
    if value >= 1e5:
        return f"PKR {value / 1e5:.2f} lac"
    return f"PKR {value:,.0f}"
