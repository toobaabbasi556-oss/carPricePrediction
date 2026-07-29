"""CLI for the two-stage scraper (SPEC §2). ``python -m pkcar.scrape <command>``.

Commands:
  robots         Fetch/parse robots.txt; print disallowed paths + Crawl-delay (rule 1).
  stage1-sample  Fetch ONE search page and print the parsed card schema (review gate).
  stage1         Full Stage 1 crawl (gated: run only after sample approval).
  stage2-sample  Fetch ONE detail page and print every field found (SPEC §2.2 gate).
  stage2         Full Stage 2 crawl (gated: run only after detail-page approval).

Politeness (delay, single worker, robots checks, request logging) is enforced by the
shared :class:`PoliteFetcher`; nothing here adds stealth/evasion (rule 7).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ..config import load_config, resolve_path
from . import robots
from .client import BlockedError, DisallowedByRobots, PoliteFetcher
from .stage1_list import (
    CARD_FIELDS,
    build_search_url,
    crawl_stage1,
    dedup_by_listing_id,
    parse_list_page,
)
from .stage2_detail import DETAIL_FIELDS, crawl_stage2, parse_detail_page
from .storage import CheckpointStore


def _build_fetcher(cfg: dict) -> tuple[PoliteFetcher, robots.RobotsRules]:
    scrape = cfg["scrape"]
    ua = scrape["user_agent"]
    rules = robots.load_rules(scrape["robots_url"], ua)
    delay = rules.crawl_delay or scrape["default_crawl_delay_sec"]
    fetcher = PoliteFetcher(
        user_agent=ua,
        delay_sec=delay,
        timeout_sec=scrape["request_timeout_sec"],
        max_retries=scrape["max_retries"],
        rules=rules,
    )
    return fetcher, rules


def cmd_robots(cfg: dict) -> int:
    scrape = cfg["scrape"]
    rules = robots.load_rules(scrape["robots_url"], scrape["user_agent"])
    print(f"Matched robots.txt group: {rules.user_agent_pattern!r}")
    default_delay = scrape["default_crawl_delay_sec"]
    print(f"Crawl-delay: {rules.crawl_delay} (default {default_delay}s if none stated)")
    print("Disallowed paths:")
    for path in rules.disallow:
        print(f"  {path!r}")
    allowed = rules.can_fetch("/used-cars/")
    print(f"\ncan_fetch('/used-cars/'): {allowed}")
    if not allowed:
        print("STOP: /used-cars/ is disallowed by robots.txt — escalate (SPEC §2.1 rule 1).")
        return 2
    return 0


def cmd_stage1_sample(cfg: dict) -> int:
    scrape = cfg["scrape"]
    fetcher, _ = _build_fetcher(cfg)
    city = scrape["cities"][0]
    url = build_search_url(scrape["search_url"], city, page=1)
    resp = fetcher.get(url)
    records = dedup_by_listing_id(parse_list_page(resp.text, scrape["base_url"]))
    print(f"Fetched {url}\nParsed {len(records)} cards. Schema: {CARD_FIELDS}")
    for rec in records[:3]:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    return 0


def cmd_stage2_sample(cfg: dict) -> int:
    """SPEC §2.2 gate: fetch ONE detail page, print every field + note gaps, then STOP."""
    scrape = cfg["scrape"]
    fetcher, _ = _build_fetcher(cfg)
    city = scrape["cities"][0]
    listing = dedup_by_listing_id(
        parse_list_page(fetcher.get(build_search_url(scrape["search_url"], city)).text,
                        scrape["base_url"])
    )
    if not listing:
        print("No listings found on sample page; cannot pick a detail URL.")
        return 1
    card = listing[0]
    detail_url = str(card["detail_url"])
    resp = fetcher.get(detail_url)
    detail = parse_detail_page(resp.text)

    print(f"Fetched detail page: {detail_url}")
    print(f"\nCARD (Stage 1):\n{json.dumps(card, ensure_ascii=False, indent=2)}")
    print(f"\nDETAIL (Stage 2) — DETAIL_FIELDS {DETAIL_FIELDS}:")
    print(json.dumps(detail, ensure_ascii=False, indent=2))

    missing = [k for k, v in detail.items() if v in (None, "", [])]
    print(f"\nMissing/empty detail fields: {missing or 'none'}")

    # Cross-check overlapping signals between card and detail (SPEC §2.2).
    print("\nCard-vs-detail consistency checks:")
    print(f"  engine_cc:   card={card.get('engine_cc')}  detail={detail.get('engine_cc')}")
    print(f"  city vs registration_city: card_city={card.get('city')} "
          f"registration_city={detail.get('registration_city')}")
    print(f"  price: card_price_pkr={card.get('price_pkr')} "
          f"canonical_price_pkr={detail.get('canonical_price_pkr')}")
    print("\nSTOP for review (SPEC §2.2).")
    return 0


def cmd_stage1_full(cfg: dict) -> int:
    """Full Stage 1 crawl -> listings_index.parquet (resumable, checkpointed)."""
    scrape = cfg["scrape"]
    paths = cfg["paths"]
    fetcher, _ = _build_fetcher(cfg)
    store = CheckpointStore(
        resolve_path(paths["interim_dir"]) / "stage1_records.jsonl",
        resolve_path(paths["interim_dir"]) / "stage1_pages_done.txt",
    )
    records = crawl_stage1(
        fetcher=fetcher,
        search_url=scrape["search_url"],
        base_url=scrape["base_url"],
        cities=scrape["cities"],
        max_items=scrape["max_items"],
        store=store,
    )
    out = store.to_parquet(resolve_path(paths["listings_index"]), dedup_key="listing_id")
    print(f"Stage 1 complete: {len(records)} unique listings -> {out}")
    return 0


def cmd_stage2_full(cfg: dict) -> int:
    """Full Stage 2 detail crawl -> listings_full.parquet (resumable, checkpointed)."""
    import pandas as pd

    paths = cfg["paths"]
    fetcher, _ = _build_fetcher(cfg)
    index_path = resolve_path(paths["listings_index"])
    if not index_path.exists():
        print(f"Stage 1 index not found at {index_path}. Run 'stage1' first.")
        return 1
    index_records = pd.read_parquet(index_path).to_dict("records")
    store = CheckpointStore(
        resolve_path(paths["interim_dir"]) / "stage2_records.jsonl",
        resolve_path(paths["interim_dir"]) / "stage2_ids_done.txt",
    )
    crawl_stage2(
        fetcher=fetcher,
        index_records=index_records,
        store=store,
        checkpoint_every=scrape_checkpoint_every(cfg),
    )
    out = store.to_parquet(resolve_path(paths["listings_full"]), dedup_key="listing_id")
    print(f"Stage 2 complete: {len(store.load_records())} detailed listings -> {out}")
    return 0


def scrape_checkpoint_every(cfg: dict) -> int:
    return int(cfg["scrape"]["detail_checkpoint_every"])


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="python -m pkcar.scrape")
    parser.add_argument(
        "command",
        nargs="?",
        default="robots",
        choices=["robots", "stage1-sample", "stage1", "stage2-sample", "stage2"],
    )
    args = parser.parse_args(argv)
    cfg = load_config()
    # Touch resolve_path so config path resolution is exercised even in sample runs.
    resolve_path(cfg["paths"]["raw_dir"]).mkdir(parents=True, exist_ok=True)

    try:
        if args.command == "robots":
            return cmd_robots(cfg)
        if args.command == "stage1-sample":
            return cmd_stage1_sample(cfg)
        if args.command == "stage2-sample":
            return cmd_stage2_sample(cfg)
        if args.command == "stage1":
            return cmd_stage1_full(cfg)
        if args.command == "stage2":
            return cmd_stage2_full(cfg)
    except DisallowedByRobots as exc:
        print(f"STOP: {exc}")
        return 2
    except BlockedError as exc:
        print(f"STOP: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
