"""Serving-layer tests (SPEC §8): payload assembly + point/interval prediction."""

from __future__ import annotations

from pkcar.data.loaders import RAW_FEATURE_COLS
from pkcar.data.split import make_group_key
from pkcar.models.boosting import build_catboost_model
from pkcar.models.intervals import ConformalIntervalModel
from pkcar.serving import predict_payload, row_from_payload
from tests.test_models import _cfg, _data


def test_row_from_payload_has_all_raw_columns() -> None:
    row = row_from_payload(
        {"make": "toyota", "model": "corolla", "year": 2018, "mileage_km": 80000}
    )
    assert list(row.columns) == list(RAW_FEATURE_COLS)
    assert row.shape == (1, len(RAW_FEATURE_COLS))
    # Missing optional fields fall back to neutral defaults, not KeyErrors.
    assert row["assembled_vs_imported"].iloc[0] == "assembled"
    assert row["fuel"].iloc[0] == "Petrol"


def test_predict_payload_returns_point_and_interval() -> None:
    df, price = _data(n=200, seed=5)
    groups = make_group_key(df)
    model = ConformalIntervalModel(build_catboost_model, alpha=0.1, config=_cfg())
    model.fit(df, price, groups)

    out = predict_payload(model, {"make": "toyota", "model": "corolla",
                                  "year": 2019, "mileage_km": 60000})
    assert out["lower_pkr"] <= out["predicted_price_pkr"] <= out["upper_pkr"]
    assert out["interval_level"] == 0.9
    assert out["currency"] == "PKR"
    assert "asking" in out["note"].lower()
