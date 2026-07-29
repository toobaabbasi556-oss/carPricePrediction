"""Residual diagnostics: residuals vs predicted, error by price band and by make (SPEC §7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _group_mape(g: pd.DataFrame) -> float:
    return float((g["abs_err"] / g["y"]).mean())


def error_by_band(
    y_true_pkr: np.ndarray, y_pred_pkr: np.ndarray, n_bands: int = 5
) -> pd.DataFrame:
    """MAE + MAPE per price quantile band (SPEC §7). Reveals where the model struggles."""
    df = pd.DataFrame({"y": y_true_pkr, "yhat": y_pred_pkr})
    df["band"] = pd.qcut(df["y"], q=n_bands, duplicates="drop")
    df["abs_err"] = (df["yhat"] - df["y"]).abs()
    grouped = df.groupby("band", observed=True)
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "mae": grouped["abs_err"].mean(),
            "mape": grouped.apply(_group_mape, include_groups=False),
        }
    )


def error_by_make(
    makes: pd.Series, y_true_pkr: np.ndarray, y_pred_pkr: np.ndarray, top: int = 12
) -> pd.DataFrame:
    """MAE + MAPE for the most common makes (SPEC §7)."""
    df = pd.DataFrame({"make": np.asarray(makes), "y": y_true_pkr, "yhat": y_pred_pkr})
    df["abs_err"] = (df["yhat"] - df["y"]).abs()
    keep = df["make"].value_counts().head(top).index
    df = df[df["make"].isin(keep)]
    grouped = df.groupby("make", observed=True)
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mae": grouped["abs_err"].mean(),
            "mape": grouped.apply(_group_mape, include_groups=False),
        }
    )
    return out.sort_values("mape", ascending=False)


def save_residuals_vs_predicted(
    y_true_pkr: np.ndarray, y_pred_pkr: np.ndarray, path: str | Path
) -> Path:
    """Residuals vs predicted on the log scale (SPEC §7); log tames the price skew."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred_log = np.log(y_pred_pkr)
    resid_log = np.log(y_true_pkr) - pred_log
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pred_log, resid_log, s=6, alpha=0.3, color="#2471a3")
    ax.axhline(0, color="#c0392b", lw=1)
    ax.set(xlabel="predicted log(price)", ylabel="residual log(price)",
           title="Residuals vs predicted (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
