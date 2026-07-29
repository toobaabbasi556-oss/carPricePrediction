"""Leakage checks (SPEC §5): fail if any feature encodes the target.

The guard is deliberately model-free and cheap so it can run in CI without the raw
dataset. It measures how well each candidate feature *alone* reconstructs the target:

- numeric feature -> absolute Pearson correlation with ``log(target)``;
- categorical feature -> correlation ratio (eta) of ``log(target)`` across its levels,
  which catches a category that is a near-unique key or a target-encoded column.

Any feature scoring above ``threshold`` is flagged. ``canonical_price_pkr`` (the
detail-page price, ~equal to the target) is the canonical positive control.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_corr(x: pd.Series, y: np.ndarray) -> float:
    """Absolute Spearman (rank) correlation.

    Rank-based so it is invariant to monotonic transforms: a column that equals the
    target — or the target's log — scores ~1.0 regardless of raw/log framing, while
    genuine noisy features stay well below the guard threshold.
    """
    xv = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(xv) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(xv[mask]) == 0:
        return 0.0
    xr = pd.Series(xv[mask]).rank().to_numpy()
    yr = pd.Series(y[mask]).rank().to_numpy()
    return float(abs(np.corrcoef(xr, yr)[0, 1]))


def _categorical_eta(x: pd.Series, y: np.ndarray) -> float:
    """Correlation ratio eta in [0, 1]: fraction of target variance explained by levels."""
    xv = x.astype("object")
    mask = xv.notna().to_numpy() & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    yv = y[mask]
    total_var = np.var(yv)
    if total_var == 0:
        return 0.0
    codes = pd.Series(xv[mask].to_numpy()).astype("category").cat.codes.to_numpy()
    between = 0.0
    grand = yv.mean()
    for c in np.unique(codes):
        g = yv[codes == c]
        between += g.size * (g.mean() - grand) ** 2
    return float(np.sqrt(between / (total_var * yv.size)))


def target_leakage_scores(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    log_target: bool = True,
) -> dict[str, float]:
    """Per-feature leakage score in [0, 1] (see module docstring). Higher = more leaky."""
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    if log_target:
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log(y)
    scores: dict[str, float] = {}
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            scores[col] = _numeric_corr(s, y)
        else:
            scores[col] = _categorical_eta(s, y)
    return scores


def assert_no_target_leakage(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    threshold: float = 0.98,
    log_target: bool = True,
) -> None:
    """Raise ``AssertionError`` if any feature reconstructs the target above ``threshold``.

    High cardinality alone is fine (make/model legitimately explain much variance); the
    guard only trips on a feature that is *essentially the target*, e.g. a duplicated
    price column. The default 0.98 sits in a wide safe gap: on the real data the most
    predictive legitimate feature (``model``) scores ~0.87, while the duplicated-price
    leak (``canonical_price_pkr``, equal to the target in ~89% of rows) scores ~0.999.
    """
    scores = target_leakage_scores(df, features, target, log_target=log_target)
    leaks = {c: v for c, v in scores.items() if v >= threshold}
    if leaks:
        detail = ", ".join(f"{c}={v:.4f}" for c, v in sorted(leaks.items(), key=lambda t: -t[1]))
        raise AssertionError(f"Feature(s) encode the target (>= {threshold}): {detail}")
