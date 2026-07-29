"""Serving layer (SPEC §8): build, persist, and query the prediction artifact.

The served artifact is a :class:`ConformalIntervalModel` wrapping the CatBoost pipeline,
fit on all priced listings so the demo uses every data point (held-out metrics were
already reported honestly in Phase 5/6). It returns a point estimate **and** a prediction
interval (SPEC §8). A raw user payload is assembled into the one-row frame the feature
pipeline expects, filling scrape-only fields (assembly heuristic) with neutral defaults.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config, resolve_path
from .data.loaders import RAW_FEATURE_COLS, TARGET_COL, load_listings
from .data.split import make_group_key
from .models.boosting import build_catboost_model
from .models.intervals import ConformalIntervalModel

logger = logging.getLogger("pkcar.serving")

# Neutral defaults for fields a user need not supply (scrape-derived or optional).
INPUT_DEFAULTS: dict[str, object] = {
    "city": "Lahore",
    "registration_city": "Punjab",
    "engine_cc": None,
    "fuel": "Petrol",
    "transmission": "Automatic",
    "body_type": None,
    "color": None,
    "seller_type": "Individual",
    "assembled_vs_imported": "assembled",
    "assembly_confidence": 0.3,
}

REQUIRED_FIELDS: tuple[str, ...] = ("make", "model", "year", "mileage_km")


def row_from_payload(payload: dict) -> pd.DataFrame:
    """Assemble a validated payload into the one-row raw frame the pipeline expects."""
    row = {col: payload.get(col, INPUT_DEFAULTS.get(col)) for col in RAW_FEATURE_COLS}
    return pd.DataFrame([row], columns=list(RAW_FEATURE_COLS))


def build_serving_model(config: dict | None = None) -> ConformalIntervalModel:
    """Fit the conformal CatBoost model on all priced listings (SPEC §6, §8)."""
    cfg = config or load_config()
    df = load_listings(drop_missing_target=True, config=cfg)
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = make_group_key(df)
    model = ConformalIntervalModel(build_catboost_model, config=cfg)
    model.fit(df, y, groups)
    return model


def save_serving_model(model: ConformalIntervalModel, config: dict | None = None) -> Path:
    """Persist the serving artifact to the configured path."""
    cfg = config or load_config()
    path = resolve_path(cfg["paths"]["serving_model"])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved serving model -> %s", path)
    return path


@lru_cache(maxsize=1)
def load_serving_model(config_path: str | None = None) -> ConformalIntervalModel:
    """Load (and cache) the serving artifact. Raises if it has not been built yet."""
    cfg = load_config(config_path) if config_path else load_config()
    path = resolve_path(cfg["paths"]["serving_model"])
    if not path.exists():
        raise FileNotFoundError(
            f"No serving model at {path}. Build it with `python -m pkcar.serving build`."
        )
    return joblib.load(path)


def predict_payload(model: ConformalIntervalModel, payload: dict) -> dict:
    """Point + interval for one raw payload, in raw PKR (SPEC §8)."""
    x = row_from_payload(payload)
    point, lower, upper = model.predict(x)
    return {
        "predicted_price_pkr": float(point[0]),
        "lower_pkr": float(lower[0]),
        "upper_pkr": float(upper[0]),
        "interval_level": round(1 - model.alpha, 2),
        "currency": "PKR",
        "note": "Predicted ASKING price (biased upward vs sale price; SPEC §3).",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build/persist the serving model (SPEC §8).")
    parser.add_argument("command", choices=["build"], help="build the serving artifact")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config) if args.config else load_config()
    model = build_serving_model(cfg)
    save_serving_model(model, cfg)


if __name__ == "__main__":
    main()
