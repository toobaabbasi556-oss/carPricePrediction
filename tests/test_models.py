"""Model-pipeline smoke tests (SPEC §5).

Tiny synthetic data + tiny estimators so CI is fast. Verifies each model builds, fits on
log-price inside the leakage-safe feature pipeline, and predicts finite values; and that
grouped CV runs fold-wise without error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pkcar.data.loaders import log_target
from pkcar.data.split import make_group_key
from pkcar.models.baseline import build_baseline_model
from pkcar.models.boosting import build_catboost_model, build_lightgbm_model
from pkcar.train import cross_val_pkr


def _cfg() -> dict:
    return {
        "project": {"current_year": 2026, "random_seed": 42},
        "features": {"rare_min_count": 2},
        "model": {
            "lgbm": {"n_estimators": 15, "num_leaves": 7, "verbose": -1},
            "catboost": {"iterations": 15, "depth": 3, "verbose": False},
        },
    }


def _data(n: int = 150, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    year = rng.integers(2005, 2024, n)
    mileage = rng.integers(0, 200_000, n)
    df = pd.DataFrame(
        {
            "make": rng.choice(["toyota", "honda", "suzuki"], n),
            "model": rng.choice(["corolla", "civic", "mehran", "alto"], n),
            "year": year,
            "mileage_km": mileage,
            "city": rng.choice(["Karachi", "Lahore"], n),
            "registration_city": rng.choice(["Sindh", "Punjab"], n),
            "engine_cc": rng.choice([660, 1000, 1300, 1800], n),
            "fuel": rng.choice(["Petrol", "Hybrid"], n),
            "transmission": rng.choice(["Automatic", "Manual"], n),
            "body_type": rng.choice(["Sedan", "Hatchback"], n),
            "color": rng.choice(["White", "Solid White", "Black"], n),
            "seller_type": rng.choice(["TRUSTED SELLER Dealer", "Ali Motors"], n),
            "assembled_vs_imported": rng.choice(["assembled", "imported"], n),
            "assembly_confidence": rng.choice([0.3, 0.95], n),
        }
    )
    price = np.exp(13.5 + 0.1 * (year - 2005) - 3e-6 * mileage + rng.normal(0, 0.2, n))
    return df, price


BUILDERS = {
    "baseline_lr": build_baseline_model,
    "lightgbm": build_lightgbm_model,
    "catboost": build_catboost_model,
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_model_fits_and_predicts_finite(name: str) -> None:
    df, price = _data()
    model = BUILDERS[name](_cfg())
    model.fit(df, log_target(price))
    preds = model.predict(df)
    assert preds.shape == (len(df),)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("name", list(BUILDERS))
def test_cross_val_pkr_returns_finite_metrics(name: str) -> None:
    df, price = _data()
    groups = make_group_key(df)
    out = cross_val_pkr(BUILDERS[name], df, price, groups, n_splits=3, config=_cfg())
    for key in ("cv_mae", "cv_rmse", "cv_mape"):
        assert key in out and np.isfinite(out[key]) and out[key] > 0
