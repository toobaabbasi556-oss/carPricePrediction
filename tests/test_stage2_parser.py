"""Stage 2 detail-page parser tests (SPEC §2.2, §4).

Validates :func:`parse_detail_page` and the assembled-vs-imported heuristic against a
saved HTML fixture of a real (trimmed) PakWheels detail page. NO live requests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pkcar.scrape.stage2_detail import (
    DETAIL_FIELDS,
    derive_assembled_vs_imported,
    parse_detail_page,
)

FIXTURE = Path(__file__).parent / "fixtures" / "detail_page_sample.html"


@pytest.fixture(scope="module")
def detail() -> dict[str, object]:
    return parse_detail_page(FIXTURE.read_text(encoding="utf-8"), scrape_date="2026-07-11")


def test_has_full_detail_schema(detail: dict[str, object]) -> None:
    assert set(detail.keys()) == set(DETAIL_FIELDS)


def test_spec_fields(detail: dict[str, object]) -> None:
    assert detail["registration_city"] == "Karachi"
    assert detail["color"] == "Solid White"
    assert detail["body_type"] == "Hatchback"
    assert detail["engine_cc"] == 660
    assert detail["seller_type"] == "Dealer"


def test_canonical_price_parsed(detail: dict[str, object]) -> None:
    assert detail["canonical_price_pkr"] == 2_865_000.0  # "PKR 28.65 lacs"


def test_assembly_from_clean_field_high_confidence(detail: dict[str, object]) -> None:
    assert detail["assembled_vs_imported"] == "assembled"
    assert detail["assembly_confidence"] == 0.95


def test_features_deduped_and_ordered(detail: dict[str, object]) -> None:
    features = detail["features"]
    assert isinstance(features, list)
    assert features[0] == "Infotainment System"
    assert "ABS" in features
    assert len(features) == len(set(features))  # no duplicates
    assert len(features) == 10


@pytest.mark.parametrize(
    ("assembly", "title", "description", "expected_label", "min_conf"),
    [
        ("Local", "Suzuki Alto 2022", "", "assembled", 0.9),
        ("Imported", "Toyota Aqua 2019", "", "imported", 0.9),
        # No clean field -> infer from text signals, lower confidence.
        (None, "Honda Vezel 2018 imported", "fresh auction sheet", "imported", 0.4),
        (None, "Suzuki Cultus 2020", "family used car", "assembled", 0.0),
    ],
)
def test_assembled_vs_imported_heuristic(
    assembly: str | None,
    title: str,
    description: str,
    expected_label: str,
    min_conf: float,
) -> None:
    label, conf = derive_assembled_vs_imported(assembly, title, description)
    assert label == expected_label
    assert 0.0 <= conf <= 1.0
    assert conf >= min_conf
