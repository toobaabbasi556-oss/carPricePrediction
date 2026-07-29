"""Stage 1 — search/list pass (SPEC §2.2).

Parse card fields (``listing_id``, ``detail_url``, ``make``, ``model``, ``year``,
``price_pkr``, ``mileage_km``, ``city``, ``engine_cc``, ``fuel``, ``transmission``)
from a PakWheels search-results page, and dedup by ``listing_id``.

The page parser (:func:`parse_list_page`) is pure — it takes HTML and returns records,
with no network access — so it can be unit-tested against a saved fixture.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from requests import RequestException

from .client import BlockedError, PoliteFetcher
from .storage import CheckpointStore

logger = logging.getLogger("pkcar.scrape")

# The Stage 1 card schema (SPEC §2.2).
CARD_FIELDS: tuple[str, ...] = (
    "listing_id",
    "detail_url",
    "make",
    "model",
    "year",
    "price_pkr",
    "mileage_km",
    "city",
    "engine_cc",
    "fuel",
    "transmission",
)

_FUELS = {"petrol", "diesel", "hybrid", "electric", "cng", "lpg", "phev"}
_TRANSMISSIONS = {"automatic", "manual"}

# /used-cars/{make}-{model}-{year}-for-sale-in-{city}-{id}
_DETAIL_RE = re.compile(
    r"/used-cars/(?P<slug>.+?)-for-sale-in-(?P<city>[a-z0-9-]+?)-(?P<id>\d+)/?$"
)
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def build_search_url(search_url: str, city: str, page: int = 1) -> str:
    """Build a city-scoped search URL. ``search_url`` ends with ``/used-cars/search/-/``."""
    base = search_url if search_url.endswith("/") else search_url + "/"
    url = f"{base}ct_{city.strip().lower()}/"
    return f"{url}?page={page}" if page and page > 1 else url


def parse_price_pkr(text: str) -> float | None:
    """Parse a PakWheels price string ('PKR 28.65 lacs', 'PKR 1.35 crore') into raw PKR."""
    if not text:
        return None
    low = text.lower()
    m = re.search(r"([\d]+(?:[\d,]*\.?\d+)?)", low.replace(",", ""))
    if not m:
        return None  # e.g. "Call for price"
    value = float(m.group(1))
    if "crore" in low:
        value *= 10_000_000
    elif "lac" in low or "lakh" in low:
        value *= 100_000
    return value


def _first_int(text: str) -> int | None:
    m = re.search(r"[\d,]+", text)
    return int(m.group(0).replace(",", "")) if m else None


def _parse_detail_url(href: str) -> dict[str, object] | None:
    """Extract listing_id / make / model / year / city from a detail-URL path."""
    m = _DETAIL_RE.search(href)
    if not m:
        return None
    tokens = [t for t in m.group("slug").split("-") if t]
    year: int | None = None
    year_idx: int | None = None
    for i in range(len(tokens) - 1, -1, -1):  # year is the last 4-digit token
        if _YEAR_RE.match(tokens[i]):
            year, year_idx = int(tokens[i]), i
            break
    make = tokens[0] if tokens else None
    model_tokens = tokens[1:year_idx] if year_idx is not None else tokens[1:]
    model = "-".join(model_tokens) or None
    return {
        "listing_id": m.group("id"),
        "make": make,
        "model": model,
        "year": year,
        "url_city": m.group("city"),
    }


def _parse_info_2(values: list[str]) -> dict[str, object]:
    """Content-match the info list (year, mileage, fuel, cc, transmission) — order-agnostic."""
    out: dict[str, object] = {"year": None, "mileage_km": None, "engine_cc": None,
                              "fuel": None, "transmission": None}
    for raw in values:
        val = raw.strip()
        low = val.lower()
        if "km" in low:
            out["mileage_km"] = _first_int(val)
        elif "cc" in low:
            out["engine_cc"] = _first_int(val)
        elif low in _FUELS:
            out["fuel"] = val
        elif low in _TRANSMISSIONS:
            out["transmission"] = val
        elif _YEAR_RE.match(val):
            out["year"] = int(val)
    return out


def parse_list_page(html: str, base_url: str) -> list[dict[str, object]]:
    """Parse a search-results page into card records. Pure function, no network."""
    soup = BeautifulSoup(html, "lxml")
    records: list[dict[str, object]] = []

    for card in soup.select("li.classified-listing"):
        link = card.select_one("a[href*='/used-cars/']")
        if link is None:
            continue
        href = link.get("href", "")
        parsed_url = _parse_detail_url(href)
        if parsed_url is None:
            continue  # not a real detail link (featured/promo/etc.)

        price_el = card.select_one(".price-details")
        price = parse_price_pkr(price_el.get_text(" ", strip=True)) if price_el else None

        city_el = card.select_one("ul.search-vehicle-info li")
        city = city_el.get_text(strip=True) if city_el else None

        info2 = card.select("ul.search-vehicle-info-2 li")
        info = _parse_info_2([li.get_text(strip=True) for li in info2])

        records.append(
            {
                "listing_id": parsed_url["listing_id"],
                "detail_url": urljoin(base_url, href),
                "make": parsed_url["make"],
                "model": parsed_url["model"],
                "year": info["year"] or parsed_url["year"],
                "price_pkr": price,
                "mileage_km": info["mileage_km"],
                "city": city or parsed_url["url_city"],
                "engine_cc": info["engine_cc"],
                "fuel": info["fuel"],
                "transmission": info["transmission"],
            }
        )
    return records


def dedup_by_listing_id(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Dedup card records by ``listing_id``, keeping first occurrence (SPEC §2.2)."""
    seen: set[object] = set()
    unique: list[dict[str, object]] = []
    for rec in records:
        lid = rec.get("listing_id")
        if lid in seen:
            continue
        seen.add(lid)
        unique.append(rec)
    return unique


def crawl_stage1(
    fetcher: PoliteFetcher,
    search_url: str,
    base_url: str,
    cities: list[str],
    max_items: int,
    store: CheckpointStore,
    max_pages_per_city: int = 200,
) -> list[dict[str, object]]:
    """Stage 1 list crawl: paginate each city, dedup, checkpoint, resume (SPEC §2.2).

    Fair per-city budget (~``max_items / len(cities)``) so all four cities are represented
    rather than the first city exhausting the global cap (city & registration_city are
    features, §4). Resumable via ``store``: pages already recorded are skipped. Stops a
    city when its budget is met or a page yields no cards; stops overall at ``max_items``.
    """
    collected: dict[str, dict[str, object]] = {r["listing_id"]: r for r in store.load_records()}
    per_city_cap = max(1, max_items // max(1, len(cities)))

    def _norm(city: object) -> str:
        return str(city or "").strip().lower()

    # Resume-safe per-city accounting: count what we already have by city.
    city_counts: dict[str, int] = {}
    for rec in collected.values():
        key = _norm(rec.get("city"))
        city_counts[key] = city_counts.get(key, 0) + 1

    for city in cities:
        key = _norm(city)
        for page in range(1, max_pages_per_city + 1):
            if len(collected) >= max_items:
                logger.info("Reached max_items=%s; stopping Stage 1.", max_items)
                return list(collected.values())
            if city_counts.get(key, 0) >= per_city_cap:
                logger.info("city=%s reached per-city cap %s; next city.", city, per_city_cap)
                break
            url = build_search_url(search_url, city, page)
            if store.is_page_done(url):
                continue  # resume: already collected this page
            try:
                resp = fetcher.get(url)
            except BlockedError:
                raise  # Cloudflare block -> STOP, no evasion (SPEC §2.1 rule 7)
            except RequestException as exc:
                logger.warning("skip %s page %s: %s", city, page, exc)
                continue  # transient failure on one page: skip it, keep crawling
            page_records = dedup_by_listing_id(parse_list_page(resp.text, base_url))
            new = [r for r in page_records if r["listing_id"] not in collected]
            for rec in new:
                collected[rec["listing_id"]] = rec
            city_counts[key] = city_counts.get(key, 0) + len(new)
            store.mark_page_done(url)
            store.append_records(new)
            logger.info("city=%s page=%s new=%s total=%s", city, page, len(new), len(collected))
            if not page_records:
                break  # no listings on this page -> end of city results
    return list(collected.values())
