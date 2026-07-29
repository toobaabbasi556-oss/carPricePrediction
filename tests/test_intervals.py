"""Conformal interval tests (SPEC §6, §7)."""

from __future__ import annotations

import numpy as np

from pkcar.data.split import make_group_key
from pkcar.models.boosting import build_catboost_model
from pkcar.models.intervals import ConformalIntervalModel, conformal_quantile_level
from tests.test_models import _cfg, _data


def test_conformal_quantile_level() -> None:
    # ceil((n+1)(1-alpha))/n
    assert conformal_quantile_level(99, 0.10) == np.ceil(100 * 0.9) / 99
    assert conformal_quantile_level(10, 0.10) <= 1.0  # capped


def test_intervals_bracket_point_and_cover() -> None:
    df, price = _data(n=250, seed=3)
    groups = make_group_key(df)
    n_test = 60
    Xtr, Xte = df.iloc[:-n_test], df.iloc[-n_test:]
    ytr, yte = price[:-n_test], price[-n_test:]

    conf = ConformalIntervalModel(build_catboost_model, alpha=0.1, config=_cfg())
    conf.fit(Xtr, ytr, groups.iloc[:-n_test])
    point, lower, upper = conf.predict(Xte)

    assert np.all(lower <= point) and np.all(point <= upper)
    assert np.all(upper > lower)
    coverage = float(np.mean((yte >= lower) & (yte <= upper)))
    assert coverage >= 0.7  # ~90% target; loose bound for a tiny synthetic sample
