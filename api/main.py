"""FastAPI service (SPEC §8). /predict returns a point estimate AND a prediction interval.

The model is provided via a dependency (:func:`get_model`) so tests can override it without
a real artifact on disk. In production it loads the cached conformal CatBoost model built
by ``python -m pkcar.serving build``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from pkcar.models.intervals import ConformalIntervalModel
from pkcar.serving import load_serving_model, predict_payload

app = FastAPI(
    title="PK Used-Car Valuation",
    version="0.1.0",
    description="Predicts a used-car ASKING price (biased upward vs sale price; SPEC §3).",
)


class CarFeatures(BaseModel):
    """User-facing inputs. Scrape-only fields default to neutral values (see serving)."""

    make: str = Field(..., examples=["toyota"])
    model: str = Field(..., examples=["corolla"])
    year: int = Field(..., ge=1950, le=2026, examples=[2018])
    mileage_km: float = Field(..., ge=0, examples=[85000])
    city: str | None = Field(None, examples=["Lahore"])
    registration_city: str | None = Field(None, examples=["Punjab"])
    engine_cc: int | None = Field(None, ge=0, examples=[1300])
    fuel: str | None = Field(None, examples=["Petrol"])
    transmission: str | None = Field(None, examples=["Automatic"])
    body_type: str | None = Field(None, examples=["Sedan"])
    color: str | None = Field(None, examples=["White"])
    seller_type: str | None = Field(None, examples=["Dealer"])


class Prediction(BaseModel):
    predicted_price_pkr: float
    lower_pkr: float
    upper_pkr: float
    interval_level: float
    currency: str
    note: str


def get_model() -> ConformalIntervalModel:
    """Model dependency (overridable in tests)."""
    return load_serving_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(
    features: CarFeatures,
    model: Annotated[ConformalIntervalModel, Depends(get_model)],
) -> Prediction:
    payload = {k: v for k, v in features.model_dump().items() if v is not None}
    return Prediction(**predict_payload(model, payload))
