"""Two-stage, robots-aware, single-worker PakWheels scraper (SPEC §2).

Stage 1 list pass -> data/raw/listings_index.parquet; Stage 2 detail pass ->
data/raw/listings_full.parquet. All HARD RULES in SPEC §2.1 apply.
"""

from . import robots
from .client import BlockedError, DisallowedByRobots, PoliteFetcher
from .stage1_list import (
    CARD_FIELDS,
    build_search_url,
    dedup_by_listing_id,
    parse_list_page,
    parse_price_pkr,
)
from .stage2_detail import DETAIL_FIELDS, derive_assembled_vs_imported, parse_detail_page
from .storage import CheckpointStore

__all__ = [
    "robots",
    "PoliteFetcher",
    "BlockedError",
    "DisallowedByRobots",
    "CheckpointStore",
    "CARD_FIELDS",
    "DETAIL_FIELDS",
    "build_search_url",
    "parse_list_page",
    "parse_price_pkr",
    "dedup_by_listing_id",
    "parse_detail_page",
    "derive_assembled_vs_imported",
]
