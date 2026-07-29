"""Feature-layer tests (SPEC §4, §5).

Covers the stateless derived features and — crucially — that the stateful transformers
(rare-category grouping, numeric imputation) learn only from the rows they are fit on,
so fitting inside a CV fold cannot leak validation information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pkcar.data.checks import assert_no_target_leakage
from pkcar.features.derived import (
    RARE_LABEL,
    UNKNOWN_LABEL,
    DerivedFeatures,
    NumericMedianImputer,
    RareCategoryGrouper,
    add_derived_features,
    normalize_color,
)
from pkcar.pipeline import (
    MODEL_FEATURES,
    build_feature_pipeline,
    categorical_features,
    model_features,
)


def _raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "make": rng.choice(["toyota", "honda", "suzuki"], n),
            "model": rng.choice(["corolla", "civic", "mehran"], n),
            "year": rng.integers(2005, 2024, n),
            "mileage_km": rng.integers(0, 200_000, n),
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


def test_age_and_buckets() -> None:
    df = pd.DataFrame(
        {"year": [2020, 2000], "engine_cc": [660, 2000], "color": ["Pearl White", "Red"],
         "seller_type": ["Some Dealer", "Private"]}
    )
    out = add_derived_features(df, current_year=2026)
    assert list(out["age"]) == [6, 26]
    assert list(out["engine_cc_bucket"]) == ["<=660", "1801-2500"]
    assert list(out["seller_is_dealer"]) == [1, 0]


def test_normalize_color_collapses_variants() -> None:
    s = pd.Series(["Solid White", "Pearl White", "White", "Black"])
    assert list(normalize_color(s)) == ["white", "white", "white", "black"]


def test_age_never_negative_for_future_year() -> None:
    out = add_derived_features(pd.DataFrame({"year": [2030], "engine_cc": [1000],
                                             "color": ["White"], "seller_type": ["x"]}),
                              current_year=2026)
    assert out["age"].iloc[0] == 0


def test_rare_grouper_buckets_rare_and_unseen_and_nan() -> None:
    train = pd.DataFrame({"model": ["civic"] * 20 + ["rareone"] * 2})
    grouper = RareCategoryGrouper(columns=["model"], min_count=10).fit(train)
    test = pd.DataFrame({"model": ["civic", "rareone", "neverseen", None]})
    out = grouper.transform(test)["model"].astype("string").tolist()
    assert out == ["civic", RARE_LABEL, RARE_LABEL, UNKNOWN_LABEL]


def test_numeric_imputer_uses_train_median_not_test() -> None:
    train = pd.DataFrame({"mileage_km": [10, 20, 30]})  # median 20
    imp = NumericMedianImputer(columns=["mileage_km"]).fit(train)
    test = pd.DataFrame({"mileage_km": [1000, np.nan]})  # test median would be 1000
    out = imp.transform(test)["mileage_km"].tolist()
    assert out == [1000.0, 20.0]  # NaN filled with TRAIN median, not test's


def test_derived_features_transformer_config_year() -> None:
    out = DerivedFeatures(current_year=2026).fit_transform(_raw())
    assert "age" in out and "color_norm" in out and "seller_is_dealer" in out


def test_pipeline_outputs_clean_model_frame() -> None:
    df = _raw()
    pipe = build_feature_pipeline()
    out = pipe.fit_transform(df)
    # Exactly the declared model features, in order.
    assert list(out.columns) == list(MODEL_FEATURES)
    # Categoricals are category dtype (native-categorical ready); numerics have no NaN.
    for c in categorical_features():
        assert str(out[c].dtype) == "category"
    assert not out.select_dtypes("number").isna().any().any()
    # Leakage / id columns never leak through.
    assert "canonical_price_pkr" not in out.columns
    assert "price_pkr" not in out.columns


def test_pipeline_handles_unseen_categories_at_transform() -> None:
    train = _raw(seed=1)
    pipe = build_feature_pipeline().fit(train)
    unseen = train.iloc[[0]].copy()
    unseen["make"] = "brandnew"  # never seen in train
    out = pipe.transform(unseen)
    assert out["make"].astype("string").iloc[0] == RARE_LABEL


@pytest.mark.parametrize("min_count", [1, 5, 50])
def test_pipeline_rare_min_count_param(min_count: int) -> None:
    out = build_feature_pipeline(rare_min_count=min_count).fit_transform(_raw())
    assert list(out.columns) == list(MODEL_FEATURES)


def test_engineered_features_pass_leakage_guard() -> None:
    """Feature engineering must not introduce a column that encodes the target (SPEC §5)."""
    df = _raw(n=300)
    # A realistic multiplicative target so the guard runs on meaningful signal.
    rng = np.random.default_rng(7)
    df["price_pkr"] = np.exp(
        14 - 0.05 * df["year"].rsub(2026) + rng.normal(0, 0.2, len(df))
    )
    engineered = build_feature_pipeline().fit_transform(df)
    engineered["price_pkr"] = df["price_pkr"].to_numpy()
    assert_no_target_leakage(engineered, model_features(), "price_pkr")
