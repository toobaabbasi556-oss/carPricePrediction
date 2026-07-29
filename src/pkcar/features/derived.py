"""Derived features + leakage-safe categorical handling (SPEC §4, §5).

Two kinds of transformer live here, all sklearn-compatible so they slot into a Pipeline
fit **inside** CV folds (SPEC §5):

- :class:`DerivedFeatures` — *stateless* row-wise features: ``age``, ``engine_cc_bucket``,
  a normalized ``color``, and a coarse ``seller_is_dealer`` flag (the raw ``seller_type``
  is unusable, ~4,900 dirty levels — EDA §3). Registration city is left untouched and
  distinct from listing city.
- :class:`RareCategoryGrouper` and :class:`NumericMedianImputer` — *stateful*: they learn
  frequent levels / medians on the training fold only, so unseen or rare categories at
  validation collapse to an explicit ``__rare__``/``__unknown__`` bucket rather than
  leaking test information (SPEC §4 "explicit unknown/rare-category handling", §5).
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..config import load_config

# Engine-size buckets roughly aligned to PK tax/segment bands (SPEC §4).
ENGINE_CC_BINS: tuple[float, ...] = (0, 660, 1000, 1300, 1800, 2500, float("inf"))
ENGINE_CC_LABELS: tuple[str, ...] = (
    "<=660",
    "661-1000",
    "1001-1300",
    "1301-1800",
    "1801-2500",
    "2500+",
)

RARE_LABEL = "__rare__"
UNKNOWN_LABEL = "__unknown__"


def _current_year(config: dict | None) -> int:
    cfg = config or load_config()
    return int(cfg["project"]["current_year"])


def normalize_color(series: pd.Series) -> pd.Series:
    """Collapse messy color levels to their base color (EDA §3).

    ``color`` has ~490 levels ("White" / "Solid White" / "Pearl White"). The base color
    is almost always the last token, so we keep that; rare leftovers are bucketed later.
    """
    base = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.rsplit(" ", n=1)
        .str[-1]
    )
    return base.replace("", pd.NA)


def add_derived_features(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    """Add stateless derived columns (SPEC §4). Pure; does not mutate ``df``."""
    out = df.copy()
    year = pd.to_numeric(out["year"], errors="coerce")
    out["age"] = (current_year - year).clip(lower=0)

    cc = pd.to_numeric(out["engine_cc"], errors="coerce")
    out["engine_cc_bucket"] = pd.cut(
        cc, bins=list(ENGINE_CC_BINS), labels=list(ENGINE_CC_LABELS), right=True
    ).astype("object")

    out["color_norm"] = normalize_color(out["color"])
    seller = out["seller_type"].astype("string").str.lower()
    out["seller_is_dealer"] = seller.str.contains("dealer", na=False).astype("int64")
    return out


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer adding the SPEC §4 derived columns. Config-driven ``current_year``."""

    def __init__(self, current_year: int | None = None) -> None:
        self.current_year = current_year

    def fit(self, X: pd.DataFrame, y: object = None) -> DerivedFeatures:  # noqa: ARG002
        self.current_year_ = self.current_year or _current_year(None)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return add_derived_features(X, self.current_year_)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Map rare/unseen categorical levels to ``__rare__`` and NaN to ``__unknown__``.

    Frequent levels are learned on the training fold only (``min_count``), so this is
    leakage-safe when fit inside CV (SPEC §5). Output columns are pandas ``category``
    dtype, ready for native-categorical boosters (SPEC §4).
    """

    def __init__(self, columns: list[str], min_count: int = 10) -> None:
        self.columns = columns
        self.min_count = min_count

    def fit(self, X: pd.DataFrame, y: object = None) -> RareCategoryGrouper:  # noqa: ARG002
        self.frequent_: dict[str, set[str]] = {}
        for col in self.columns:
            counts = X[col].astype("string").value_counts()
            self.frequent_[col] = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            s = out[col].astype("string")
            keep = self.frequent_[col]
            mapped = s.where(s.isin(keep), other=RARE_LABEL)
            mapped = mapped.where(s.notna(), other=UNKNOWN_LABEL)
            out[col] = mapped.astype("category")
        return out


class NumericMedianImputer(BaseEstimator, TransformerMixin):
    """Median-impute numeric columns, learning medians on the training fold (SPEC §5)."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> NumericMedianImputer:  # noqa: ARG002
        self.medians_: dict[str, float] = {}
        for col in self.columns:
            self.medians_[col] = float(pd.to_numeric(X[col], errors="coerce").median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals.fillna(self.medians_[col]).astype("float64")
        return out
