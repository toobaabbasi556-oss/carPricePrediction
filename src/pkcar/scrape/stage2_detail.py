"""Stage 2 — detail pass (SPEC §2.2).

Visit each unique ``detail_url`` **once**; pull ``registration_city``, ``color``,
``body_type``, ``assembled_vs_imported`` (heuristic + confidence), ``seller_type``,
engine capacity, feature list, and the canonical price. Checkpoint every 50 pages,
resumable. Merge onto the Stage 1 index with a ``scrape_date`` column.

NOTE — SPEC §2.2 gate: before scaling Stage 2, fetch ONE detail page, print every
field found + which are missing/inconsistent vs the card, then STOP for review. The
spec-table selectors in :func:`parse_detail_page` are finalized at that gate; the
label→field mapping below is the first-pass hypothesis to validate against a live page.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re

import pandas as pd
from bs4 import BeautifulSoup
from requests import RequestException

from ..features.assembled import derive_assembled_vs_imported
from .client import BlockedError, DisallowedByRobots, PoliteFetcher
from .stage1_list import parse_price_pkr
from .storage import CheckpointStore

__all__ = ["crawl_stage2", "derive_assembled_vs_imported", "merge_onto_index", "parse_detail_page"]

logger = logging.getLogger("pkcar.scrape")

# Detail fields we add on top of the Stage 1 card (SPEC §2.2).
DETAIL_FIELDS: tuple[str, ...] = (
    "registration_city",
    "color",
    "body_type",
    "assembled_vs_imported",
    "assembly_confidence",
    "seller_type",
    "engine_cc",
    "features",
    "canonical_price_pkr",
    "scrape_date",
)

# Normalized spec-table label -> our field name. Verify/extend at the §2.2 gate.
_LABEL_MAP = {
    "registered in": "registration_city",
    "registration city": "registration_city",
    "color": "color",
    "colour": "color",
    "body type": "body_type",
    "assembly": "assembly",  # "Local" / "Imported" when present
    "engine capacity": "engine_cc",
    "seller": "seller_type",
}

def _extract_spec_table(soup: BeautifulSoup) -> dict[str, str]:
    """Pull label→value pairs from ``#scroll_car_detail`` (verified live at the §2.2 gate).

    PakWheels renders the spec table as a flat ``<li>`` list where a label carries
    ``class="ad-data"`` and the value is the immediately following ``<li>``.
    """
    specs: dict[str, str] = {}
    lis = soup.select("#scroll_car_detail li")
    i = 0
    while i < len(lis):
        classes = lis[i].get("class") or []
        if "ad-data" in classes and i + 1 < len(lis):
            label = lis[i].get_text(strip=True).lower().rstrip(":").strip()
            specs[label] = lis[i + 1].get_text(strip=True)
            i += 2
        else:
            i += 1
    return specs


def _parse_seller_type(soup: BeautifulSoup) -> str | None:
    """Seller type from the owner block, e.g. 'Dealer: Red Dot ...' -> 'Dealer'."""
    el = soup.select_one(".owner-detail-main")
    if el is None:
        return None
    text = el.get_text(" ", strip=True)
    head = text.split(":", 1)[0].strip()
    return head or None


def parse_detail_page(html: str, scrape_date: str | None = None) -> dict[str, object]:
    """Parse one detail page into DETAIL_FIELDS. Pure function, no network.

    First-pass selectors — validated at the SPEC §2.2 one-page review gate.
    """
    soup = BeautifulSoup(html, "lxml")
    specs = _extract_spec_table(soup)

    mapped: dict[str, object] = {}
    for raw_label, value in specs.items():
        field = _LABEL_MAP.get(raw_label.strip())
        if field and field != "assembly":
            mapped[field] = value
    if "engine_cc" in mapped:
        m = re.search(r"[\d,]+", str(mapped["engine_cc"]))
        mapped["engine_cc"] = int(m.group(0).replace(",", "")) if m else None

    title_el = soup.select_one("h1")
    title = title_el.get_text(strip=True) if title_el else ""
    # Seller's Comments block (h2#scroll_seller_comments heads it) feeds the import heuristic.
    comments_heading = soup.select_one("#scroll_seller_comments")
    description = ""
    if comments_heading is not None and comments_heading.parent is not None:
        description = comments_heading.parent.get_text(" ", strip=True)

    label, confidence = derive_assembled_vs_imported(specs.get("assembly"), title, description)

    price_el = soup.select_one("div.price-box strong, div.price-box")
    canonical_price = parse_price_pkr(price_el.get_text(" ", strip=True)) if price_el else None

    # Features span several category lists; dedup while preserving first-seen order.
    feature_els = soup.select("ul.car-feature-list li, #feature_content li")
    features = list(dict.fromkeys(f.get_text(strip=True) for f in feature_els))

    return {
        "registration_city": mapped.get("registration_city"),
        "color": mapped.get("color"),
        "body_type": mapped.get("body_type"),
        "assembled_vs_imported": label,
        "assembly_confidence": confidence,
        "seller_type": _parse_seller_type(soup),
        "engine_cc": mapped.get("engine_cc"),
        "features": features,
        "canonical_price_pkr": canonical_price,
        "scrape_date": scrape_date or _dt.date.today().isoformat(),
    }


def crawl_stage2(
    fetcher: PoliteFetcher,
    index_records: list[dict[str, object]],
    store: CheckpointStore,
    checkpoint_every: int = 50,
) -> None:
    """Detail crawl: visit each detail_url once, checkpoint, resume (SPEC §2.2).

    NOT run during review — gated on the §2.2 one-page approval, then explicit go.
    """
    scrape_date = _dt.date.today().isoformat()
    since_checkpoint = 0
    for rec in index_records:
        listing_id = str(rec["listing_id"])
        if store.is_id_done(listing_id):
            continue  # resume: already fetched
        try:
            resp = fetcher.get(str(rec["detail_url"]))
        except BlockedError:
            raise  # Cloudflare block -> STOP the whole crawl, no evasion (SPEC §2.1 rule 7)
        except (RequestException, DisallowedByRobots) as exc:
            # Transient 5xx (post-retry), 404, or a disallowed url: skip this listing only.
            logger.warning("skip %s: %s", listing_id, exc)
            store.record_failure(listing_id, str(exc)[:200])
            continue
        detail = parse_detail_page(resp.text, scrape_date=scrape_date)
        store.append_records([{**rec, **detail}])
        store.mark_id_done(listing_id)
        since_checkpoint += 1
        if since_checkpoint >= checkpoint_every:
            logger.info("Checkpoint: %s details collected so far.", checkpoint_every)
            since_checkpoint = 0


def merge_onto_index(
    index_path: str, detail_records: list[dict[str, object]], out_path: str
) -> pd.DataFrame:
    """Merge Stage 2 detail records onto the Stage 1 index -> listings_full (SPEC §2.2)."""
    index = pd.read_parquet(index_path)
    details = pd.DataFrame(detail_records)
    merged = index.merge(details, on="listing_id", how="left", suffixes=("", "_detail"))
    merged.to_parquet(out_path, index=False)
    return merged
