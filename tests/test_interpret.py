"""SHAP + residual-diagnostic tests (SPEC §6, §7)."""

from __future__ import annotations

import numpy as np

from pkcar.data.loaders import log_target
from pkcar.evaluation.residuals import error_by_band, error_by_make
from pkcar.evaluation.shap_plots import compute_shap_values, mean_abs_shap
from pkcar.models.boosting import build_catboost_model
from pkcar.pipeline import model_features
from tests.test_models import _cfg, _data


def test_shap_values_shape_and_additivity() -> None:
    df, price = _data(n=200, seed=1)
    model = build_catboost_model(_cfg())
    model.fit(df, log_target(price))

    shap_values, xt, expected = compute_shap_values(model, df)
    assert shap_values.shape == (len(df), len(model_features()))
    assert list(xt.columns) == model_features()

    # SHAP additivity: base value + sum(contributions) == model prediction (log scale).
    reconstructed = expected + shap_values.sum(axis=1)
    preds = model.predict(df)
    assert np.allclose(reconstructed, preds, atol=1e-4)


def test_mean_abs_shap_ranks_features() -> None:
    df, price = _data(n=200, seed=2)
    model = build_catboost_model(_cfg())
    model.fit(df, log_target(price))
    shap_values, xt, _ = compute_shap_values(model, df)
    ranking = mean_abs_shap(shap_values, xt)
    assert len(ranking) == len(model_features())
    assert (ranking >= 0).all()
    assert ranking.is_monotonic_decreasing


def test_error_by_band_and_make() -> None:
    df, price = _data(n=200, seed=4)
    pred = price * 1.1  # 10% high everywhere
    band = error_by_band(price, pred, n_bands=4)
    assert {"n", "mae", "mape"}.issubset(band.columns)
    assert np.allclose(band["mape"].to_numpy(), 0.1, atol=1e-9)

    make = error_by_make(df["make"], price, pred)
    assert {"n", "mae", "mape"}.issubset(make.columns)
    assert (make["n"] > 0).all()
