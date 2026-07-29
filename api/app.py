"""Streamlit demo front end over the FastAPI /predict endpoint (SPEC §8).

If ``PKCAR_API_URL`` is set, the app POSTs to that FastAPI ``/predict``; otherwise it loads
the serving model in-process (single-container HF Spaces / Render deploy). Either way it
shows the point estimate and the prediction interval, with the asking-price caveat (§3).
"""

from __future__ import annotations

import os

import streamlit as st

from pkcar.evaluation.metrics import format_pkr

API_URL = os.environ.get("PKCAR_API_URL")


def _predict(payload: dict) -> dict:
    if API_URL:
        import requests

        resp = requests.post(f"{API_URL.rstrip('/')}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    # In-process fallback (no separate API server needed).
    from pkcar.serving import load_serving_model, predict_payload

    return predict_payload(load_serving_model(), payload)


st.set_page_config(page_title="PK Used-Car Valuation", page_icon="🚗")
st.title("🚗 PK Used-Car Valuation")
st.caption(
    "Predicts a used-car **asking price** on PakWheels. Asking prices are seller-set and "
    "biased upward vs actual sale prices (SPEC §3) — treat this as a listing estimate."
)

col1, col2 = st.columns(2)
with col1:
    make = st.text_input("Make", "toyota")
    model = st.text_input("Model", "corolla")
    year = st.number_input("Year", min_value=1950, max_value=2026, value=2018)
    mileage_km = st.number_input("Mileage (km)", min_value=0, value=85_000, step=5_000)
    engine_cc = st.number_input("Engine (cc)", min_value=0, value=1300, step=100)
with col2:
    city = st.text_input("Listing city", "Lahore")
    registration_city = st.text_input("Registration city", "Punjab")
    fuel = st.selectbox("Fuel", ["Petrol", "Hybrid", "Diesel", "CNG", "Electric"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
    seller_type = st.selectbox("Seller", ["Dealer", "Individual"])

if st.button("Estimate price", type="primary"):
    payload = {
        "make": make.strip().lower(),
        "model": model.strip().lower(),
        "year": int(year),
        "mileage_km": float(mileage_km),
        "city": city,
        "registration_city": registration_city,
        "engine_cc": int(engine_cc),
        "fuel": fuel,
        "transmission": transmission,
        "seller_type": seller_type,
    }
    try:
        result = _predict(payload)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the demo user
        st.error(f"Prediction failed: {exc}")
    else:
        st.metric("Estimated asking price", format_pkr(result["predicted_price_pkr"]))
        level = int(round(100 * result["interval_level"]))
        st.write(
            f"**{level}% prediction interval:** {format_pkr(result['lower_pkr'])} "
            f"— {format_pkr(result['upper_pkr'])}"
        )
        st.caption(result.get("note", ""))
