# CLAUDE.md — PK Used-Car Valuation

Standing conventions for this repo. **SPEC.md is the full source of truth**; this
file is the always-on rules distilled from SPEC §2 and §5. When in doubt, read SPEC.md.

## Data collection (SPEC §2 — non-negotiable)
1. **robots.txt FIRST.** Fetch/parse `https://www.pakwheels.com/robots.txt` before any
   scrape runs; print disallowed paths + Crawl-delay. If `/used-cars/` is disallowed, **STOP**.
2. Respect `Crawl-delay`; default to **3s** between requests if none is stated.
3. One worker, **no concurrency**. Descriptive User-Agent (bot name + purpose).
4. Config-driven: cities, `max_items`, delay live in `config/config.yaml` — never hardcoded.
5. Checkpoint frequently and be resumable (skip already-collected pages/ids).
6. Log every request: url, status, timestamp.
7. **No stealth/evasion.** On repeated 403/Cloudflare blocks, STOP and report.
8. Raw data stays **local** and gitignored. Never publish the raw dataset.
9. Two-stage only: Stage 1 list → `listings_index.parquet`; Stage 2 detail → `listings_full.parquet`.
   Visit each `detail_url` once. `assembled_vs_imported` is heuristic + a confidence flag.

## Modeling (SPEC §5 — non-negotiable)
- **Target is `log(price_pkr)`.** Report ALL metrics back in **raw PKR** (MAE/RMSE/MAPE).
- **NO leakage:** every transform (encode/scale/impute) lives in an `sklearn.Pipeline`
  and is fit **inside** CV folds. Never fit on the full dataset.
- Cross-validation is **temporal/grouped-aware**. No naive KFold.
- A pytest **leakage guard** must fail if any feature encodes the target.

## Engineering (SPEC §8)
- Config-driven everywhere: no hardcoded paths/params in `.py` files; read `config/config.yaml`.
- `src/` layout, typed signatures, docstrings. `ruff` clean. `pytest` + CI must pass.
- Prices are `float`, never `str`. Registration city is a first-class feature, kept
  distinct from listing city.
