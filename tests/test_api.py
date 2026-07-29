"""FastAPI /predict tests (SPEC §8). Model injected via dependency override — no artifact."""

from __future__ import annotations

import numpy as np
from api.main import app, get_model
from fastapi.testclient import TestClient


class _StubModel:
    alpha = 0.1

    def predict(self, X):  # noqa: ANN001, ARG002
        return (np.array([3_000_000.0]), np.array([2_100_000.0]), np.array([4_300_000.0]))


app.dependency_overrides[get_model] = lambda: _StubModel()
client = TestClient(app)


def test_health() -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_point_and_interval() -> None:
    r = client.post("/predict", json={"make": "toyota", "model": "corolla",
                                      "year": 2018, "mileage_km": 85000})
    assert r.status_code == 200
    body = r.json()
    assert body["lower_pkr"] <= body["predicted_price_pkr"] <= body["upper_pkr"]
    assert body["interval_level"] == 0.9
    assert body["currency"] == "PKR"


def test_predict_validates_input() -> None:
    # Missing required 'make' -> 422; out-of-range year -> 422.
    assert client.post("/predict", json={"model": "civic", "year": 2018,
                                         "mileage_km": 10}).status_code == 422
    assert client.post("/predict", json={"make": "honda", "model": "civic",
                                         "year": 1800, "mileage_km": 10}).status_code == 422
