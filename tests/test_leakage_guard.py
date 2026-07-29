"""Leakage-guard tests (SPEC §5): fail if any feature encodes the target.

Runs on a small synthetic frame that mirrors the real schema, so it needs no raw
data and runs in CI. A clean feature set must pass; injecting the known leaked column
(``canonical_price_pkr``) must trip the guard — proving the test has teeth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pkcar.data.checks import assert_no_target_leakage, target_leakage_scores
from pkcar.data.loaders import LEAKAGE_COLS, TARGET_COL, feature_columns


def _synthetic(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    makes = rng.choice(["toyota", "honda", "suzuki"], n)
    models = rng.choice(["corolla", "civic", "mehran", "alto"], n)
    year = rng.integers(2005, 2024, n)
    mileage = rng.integers(0, 200_000, n)
    # Price depends on year & mileage plus noise (multiplicative -> use exp of a log form).
    log_price = 13.5 + 0.12 * (year - 2005) - 3e-6 * mileage + rng.normal(0, 0.25, n)
    price = np.exp(log_price)
    return pd.DataFrame(
        {
            "make": makes,
            "model": models,
            "year": year,
            "mileage_km": mileage,
            "city": rng.choice(["Karachi", "Lahore"], n),
            "registration_city": rng.choice(["Sindh", "Punjab", "Islamabad"], n),
            "engine_cc": rng.choice([660, 1000, 1300, 1800], n),
            "fuel": rng.choice(["Petrol", "Hybrid"], n),
            "transmission": rng.choice(["Automatic", "Manual"], n),
            "body_type": rng.choice(["Sedan", "Hatchback"], n),
            "color": rng.choice(["White", "Black", "Silver"], n),
            "seller_type": rng.choice(["Dealer", "Individual"], n),
            "assembled_vs_imported": rng.choice(["assembled", "imported"], n),
            "assembly_confidence": rng.choice([0.3, 0.95], n),
            TARGET_COL: price,
            "canonical_price_pkr": price,  # the leaked duplicate of the target
        }
    )


def test_declared_features_exclude_target_and_leakage() -> None:
    feats = feature_columns()
    assert TARGET_COL not in feats
    for leaked in LEAKAGE_COLS:
        assert leaked not in feats, f"{leaked} must not be a feature"


def test_clean_feature_set_passes() -> None:
    df = _synthetic()
    # Should not raise: legitimate features never *are* the target.
    assert_no_target_leakage(df, feature_columns(), TARGET_COL)


def test_guard_catches_injected_leak() -> None:
    df = _synthetic()
    leaked_features = feature_columns() + list(LEAKAGE_COLS)
    with pytest.raises(AssertionError, match="encode the target"):
        assert_no_target_leakage(df, leaked_features, TARGET_COL)


def test_leaked_column_scores_near_one() -> None:
    df = _synthetic()
    scores = target_leakage_scores(df, list(LEAKAGE_COLS), TARGET_COL)
    assert scores["canonical_price_pkr"] > 0.999
