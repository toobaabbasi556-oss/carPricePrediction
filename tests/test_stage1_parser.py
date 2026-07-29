"""Stage 1 list-page parser tests (SPEC §2.2).

Validates :func:`parse_list_page` against a saved HTML fixture of real (trimmed)
PakWheels cards. NO live requests — the parser is a pure function over HTML.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pkcar.scrape.stage1_list import (
    CARD_FIELDS,
    build_search_url,
    dedup_by_listing_id,
    parse_list_page,
    parse_price_pkr,
)

FIXTURE = Path(__file__).parent / "fixtures" / "list_page_sample.html"
BASE_URL = "https://www.pakwheels.com"


@pytest.fixture(scope="module")
def records() -> list[dict[str, object]]:
    html = FIXTURE.read_text(encoding="utf-8")
    return parse_list_page(html, BASE_URL)


def test_parses_all_cards(records: list[dict[str, object]]) -> None:
    assert len(records) == 3


def test_every_record_has_full_schema(records: list[dict[str, object]]) -> None:
    for rec in records:
        assert set(rec.keys()) == set(CARD_FIELDS)


def test_listing_ids_and_urls(records: list[dict[str, object]]) -> None:
    by_id = {r["listing_id"]: r for r in records}
    assert set(by_id) == {"11587555", "11704950", "11584902"}
    assert by_id["11587555"]["detail_url"] == (
        "https://www.pakwheels.com/used-cars/suzuki-alto-2022-for-sale-in-karachi-11587555"
    )


def test_make_model_year_from_url(records: list[dict[str, object]]) -> None:
    by_id = {r["listing_id"]: r for r in records}
    alto = by_id["11587555"]
    assert (alto["make"], alto["model"], alto["year"]) == ("suzuki", "alto", 2022)
    # Multi-token model with a trailing number is preserved.
    shark = by_id["11704950"]
    assert (shark["make"], shark["model"]) == ("byd", "shark-6")


def test_card_field_values(records: list[dict[str, object]]) -> None:
    alto = next(r for r in records if r["listing_id"] == "11587555")
    assert alto["price_pkr"] == 2_865_000.0  # "PKR 28.65 lacs"
    assert alto["mileage_km"] == 70_000
    assert alto["engine_cc"] == 660
    assert alto["fuel"] == "Petrol"
    assert alto["transmission"] == "Automatic"
    assert alto["city"] == "Karachi"


def test_phev_fuel_and_crore_price(records: list[dict[str, object]]) -> None:
    shark = next(r for r in records if r["listing_id"] == "11704950")
    assert shark["fuel"] == "PHEV"  # non-obvious fuel label must survive parsing
    assert shark["price_pkr"] == 21_100_000.0  # "PKR 2.11 crore"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("PKR 28.65 lacs", 2_865_000.0),
        ("PKR 1.4 crore", 14_000_000.0),
        ("PKR 95 lacs", 9_500_000.0),
        ("Rs 12,50,000", 1_250_000.0),
        ("Call for price", None),
        ("", None),
    ],
)
def test_parse_price_pkr(text: str, expected: float | None) -> None:
    assert parse_price_pkr(text) == expected


def test_dedup_by_listing_id() -> None:
    dupes = [
        {"listing_id": "1", "make": "a"},
        {"listing_id": "1", "make": "b"},
        {"listing_id": "2", "make": "c"},
    ]
    out = dedup_by_listing_id(dupes)
    assert [r["listing_id"] for r in out] == ["1", "2"]
    assert out[0]["make"] == "a"  # first occurrence kept


def test_build_search_url() -> None:
    base = "https://www.pakwheels.com/used-cars/search/-/"
    assert build_search_url(base, "Karachi", 1) == base + "ct_karachi/"
    assert build_search_url(base, "Lahore", 3) == base + "ct_lahore/?page=3"
