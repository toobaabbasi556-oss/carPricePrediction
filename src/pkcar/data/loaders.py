"""Loaders for raw/processed parquet + the column-role contract (SPEC §2.2, §4, §5).

This module is the single place that declares which columns are the target, which
are identifiers, which are **leakage** columns that must never be fed to a model,
and which are legitimate raw features (SPEC §4).

Note: ``price_pkr`` is a seller-set **asking price**, biased upward (SPEC §3). The
model predicts asking price, not transaction price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import load_config, resolve_path

# --- Column-role contract (SPEC §4, §5) -------------------------------------

TARGET_COL = "price_pkr"

# Identifiers / provenance — not features. ``scrape_date`` is crawl provenance only:
# it holds just a few values reflecting when our resumable crawler ran (07-12/14/15),
# NOT when a listing was posted, so it is not a valid temporal axis (SPEC §11).
ID_COLS: tuple[str, ...] = ("listing_id", "detail_url", "scrape_date")

# Columns derived from / encoding the target — MUST be excluded from features.
# ``canonical_price_pkr`` (detail-page price) equals ``price_pkr`` in ~89% of rows
# (log-corr 0.996); feeding it would leak the target. The leakage guard enforces this.
LEAKAGE_COLS: tuple[str, ...] = ("canonical_price_pkr",)

# Raw features (SPEC §4). Registration city is first-class and kept distinct from
# listing city. Derived features (age, engine_cc buckets) are added in the feature
# layer (SPEC §4, Phase 4); assembled_vs_imported was derived during the scrape.
RAW_FEATURE_COLS: tuple[str, ...] = (
    "make",
    "model",
    "year",
    "mileage_km",
    "city",
    "registration_city",
    "engine_cc",
    "fuel",
    "transmission",
    "body_type",
    "color",
    "seller_type",
    "assembled_vs_imported",
    "assembly_confidence",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "make",
    "model",
    "city",
    "registration_city",
    "fuel",
    "transmission",
    "body_type",
    "color",
    "seller_type",
    "assembled_vs_imported",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "year",
    "mileage_km",
    "engine_cc",
    "assembly_confidence",
)


def feature_columns() -> list[str]:
    """Raw feature columns (SPEC §4). Excludes target, ids, and leakage columns."""
    return list(RAW_FEATURE_COLS)


def target_column() -> str:
    """Name of the raw-PKR target (trained on its log, SPEC §5)."""
    return TARGET_COL


def load_listings(
    drop_missing_target: bool = True,
    config: dict | None = None,
) -> pd.DataFrame:
    """Load ``listings_full.parquet`` and apply minimal, non-leaky cleaning.

    - Drops rows with no ``price_pkr`` (114 "Call for price" listings, SPEC §3) when
      ``drop_missing_target`` is set — they carry no target and cannot be modeled.
    - Coerces obvious dtypes. Does **not** impute, encode, or scale — those live inside
      the CV-fold pipeline (SPEC §5) to avoid leakage.

    Args:
        drop_missing_target: drop rows whose target is missing (default True).
        config: parsed config; loaded from ``config/config.yaml`` if omitted.

    Returns:
        A DataFrame with the raw schema (features + target + ids + leakage cols kept
        so callers can inspect them; use :func:`feature_columns` to select features).
    """
    cfg = config or load_config()
    path = resolve_path(cfg["paths"]["listings_full"])
    df = pd.read_parquet(path)

    df["price_pkr"] = pd.to_numeric(df["price_pkr"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    if drop_missing_target:
        df = df.loc[df[TARGET_COL].notna()].reset_index(drop=True)
    return df


def log_target(prices: pd.Series | np.ndarray) -> np.ndarray:
    """``log(price_pkr)`` — the modeling target (SPEC §5). Metrics report back in PKR."""
    return np.log(np.asarray(prices, dtype=float))


def inverse_log_target(log_prices: pd.Series | np.ndarray) -> np.ndarray:
    """Invert :func:`log_target` back to raw PKR for metric reporting (SPEC §5, §7)."""
    return np.exp(np.asarray(log_prices, dtype=float))
