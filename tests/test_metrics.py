"""Metric tests (SPEC §7): PKR point metrics and interval metrics, with log inversion."""

from __future__ import annotations

import numpy as np

from pkcar.data.loaders import log_target
from pkcar.evaluation.metrics import (
    format_pkr,
    interval_metrics_pkr,
    metrics_from_log_predictions,
    regression_metrics_pkr,
)


def test_regression_metrics_known_values() -> None:
    m = regression_metrics_pkr(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert m["mae"] == 15.0
    assert abs(m["rmse"] - np.sqrt(250.0)) < 1e-9
    assert abs(m["mape"] - 0.10) < 1e-9


def test_metrics_from_log_predictions_perfect() -> None:
    y = np.array([100_000.0, 3_000_000.0])
    m = metrics_from_log_predictions(log_target(y), log_target(y))
    assert m["mae"] < 1e-6 and m["mape"] < 1e-9


def test_metrics_reported_in_pkr_not_log() -> None:
    # A fixed log error must translate to a PKR-scale MAE, not a ~0.1 log-scale number.
    y = np.array([1_000_000.0, 2_000_000.0])
    pred_log = log_target(y) + 0.1
    m = metrics_from_log_predictions(log_target(y), pred_log)
    assert m["mae"] > 50_000  # clearly in rupees, not log units


def test_interval_metrics() -> None:
    yt = np.array([100.0, 200.0, 300.0])
    lo = np.array([90.0, 250.0, 280.0])
    hi = np.array([110.0, 260.0, 320.0])
    m = interval_metrics_pkr(yt, lo, hi)
    assert abs(m["coverage"] - 2 / 3) < 1e-9  # 1st and 3rd covered, 2nd not
    assert abs(m["mean_width"] - 70 / 3) < 1e-9  # widths [20, 10, 40]


def test_format_pkr() -> None:
    assert "crore" in format_pkr(15_000_000)
    assert "lac" in format_pkr(345_000)
    assert format_pkr(50_000).startswith("PKR 50,000")
