# SPEC — PK Used-Car Valuation Service

**Type:** Portfolio / CV project (not research)
**Owner:** Tooba Maqsood
**One-line goal:** A production-grade service that predicts used-car asking prices for the Pakistani market, with prediction intervals, interpretability, and a deployed API + demo.

The point of this project is to demonstrate **end-to-end ML engineering discipline** — self-collected data, leakage-free pipelines, honest evaluation, deployment, and monitoring — on a domain (the Pakistani used-car market) that a generic Kaggle-notebook candidate cannot replicate. Squeezing out the last decimal of R² is **not** the goal. Rigor and shippability are.

---

## 1. Scope

**In scope**
- Self-scraped PakWheels data (two-stage, polite, robots.txt-compliant).
- Domain feature engineering specific to Pakistan (registration city, assembled-vs-imported, etc.).
- Leakage-free modeling with honest cross-validation.
- Prediction intervals (uncertainty), not just point estimates.
- SHAP interpretability.
- Deployed FastAPI endpoint + Streamlit demo, containerised.
- Experiment tracking, tests, CI, a README that reads like a model card.

**Out of scope**
- Beating a target metric. There is no "must hit X% MAPE" requirement.
- Real-time / streaming data. One-off scrape + periodic refresh only.
- Scraping the full 200k listing universe. Capped, scoped, polite.
- Redistributing the raw scraped dataset publicly.

---

## 2. Data source & collection rules

**Source:** PakWheels used-car listings — `https://www.pakwheels.com/used-cars/search/-/`
Detail-page pattern: `/used-cars/{make}-{model}-{year}-for-sale-in-{city}-{id}`

### 2.1 Non-negotiable collection rules
1. **robots.txt first.** Before any scraping code runs, fetch and parse `https://www.pakwheels.com/robots.txt`. Print disallowed paths and any `Crawl-delay`. **If `/used-cars/` is disallowed, STOP** and escalate — do not scrape it.
2. Respect `Crawl-delay`. If none is stated, default to a **3-second** delay between requests.
3. **One worker, no concurrency.** Descriptive User-Agent identifying the bot (name + purpose).
4. Config-driven: cities, `max_items`, delay all live in `config/config.yaml`.
5. Checkpoint frequently, resumable (skip already-collected pages/ids on restart).
6. Log every request: url, status, timestamp.
7. **No stealth/evasion.** If repeated 403/Cloudflare blocks occur, STOP and report — do not add anti-detection tricks.
8. Keep raw data **local**. Do not publish the raw dataset.

### 2.2 Two-stage scrape
**Stage 1 — list pass**
- Cities: Karachi, Lahore, Islamabad, Rawalpindi. `max_items: 7000`.
- Card fields: `listing_id`, `detail_url`, `make`, `model`, `year`, `price_pkr`, `mileage_km`, `city`, `engine_cc`, `fuel`, `transmission`.
- Dedup by `listing_id` → `data/raw/listings_index.parquet`.

**Stage 2 — detail pass**
- Visit each unique `detail_url` **once**.
- Pull: `registration_city`, `color`, `body_type`, `assembled_vs_imported`, `seller_type`, engine capacity, feature list (if present), canonical price.
- Checkpoint every 50 pages, resumable.
- Merge onto the index → `data/raw/listings_full.parquet` with a `scrape_date` column.
- **Before scaling Stage 2:** fetch ONE detail page, print every field found + which are missing/inconsistent vs the card, then STOP for review.

> `assembled_vs_imported` is often not a clean field — it hides in titles/descriptions ("imported", "unregistered", auction-sheet mentions). Derive it heuristically and attach a confidence flag; do not assume a tidy column exists.

---

## 3. Critical data-quality caveat (must appear in README)

PakWheels prices are **seller-set asking prices**, not transaction prices, and they are **biased upward** — sellers commonly copy the prices of similar listings, which anchors and inflates the market. This model therefore predicts **asking price**, not sale price. State this openly. Naming the bias and its direction is a maturity signal worth more than a metric.

---

## 4. Features

**Raw:** make, model, year, mileage_km, city, registration_city, engine_cc, fuel, transmission, body_type, color, seller_type.

**Derived:**
- `age = current_year - year`
- `assembled_vs_imported` (heuristic + confidence)
- engine_cc buckets
- high-cardinality handling for make/model/variant (native categorical support, not naive one-hot)
- explicit unknown/rare-category handling

Registration city is a **first-class feature** (it materially affects price/transferability in PK) and must be kept distinct from listing city.

---

## 5. Modeling requirements

- **Target:** `log(price_pkr)`. Prices are right-skewed and multiplicative. Report all metrics back in **raw PKR**.
- **Baseline:** Linear Regression on log-price — a genuine baseline, not a co-headline.
- **Main models:** CatBoost and/or LightGBM (native high-cardinality categorical handling).
- **Cross-validation:** temporal/grouped-aware. No naive KFold — the same model recurs across rows and will leak.
- **No leakage:** every transform (encoding, scaling, imputation) lives in an `sklearn.Pipeline` and is fit **inside** CV folds. Never fit on the full dataset.
- **Leakage guard:** a pytest test that fails if any feature encodes the target.

---

## 6. Uncertainty & interpretability

- **Prediction intervals** via LightGBM quantile regression or conformal prediction. Report empirical **interval coverage** (does the 90% interval contain the truth ~90% of the time?).
- **SHAP** summary + a few dependence plots to explain price drivers.

---

## 7. Evaluation

- Point metrics in PKR: **MAE, RMSE, MAPE** (MAPE matters for prices).
- Interval metrics: coverage + mean interval width.
- Residual diagnostics: residuals vs predicted, error by price band and by make.
- Honest test protocol: a held-out set the model never touched during CV/tuning.

---

## 8. Engineering & deployment

- `src/` layout, typed signatures, docstrings.
- **Config-driven** — no hardcoded paths/params in `.py` files; everything reads `config/config.yaml`.
- **Experiment tracking:** MLflow.
- **Tests:** pytest (schema, leakage guard, parser fixtures, interval coverage sanity).
- **CI:** GitHub Actions — ruff lint + pytest on push.
- **Serving:** FastAPI `/predict` returning point estimate **and** interval; Streamlit demo front end.
- **Container:** Dockerfile.
- **Deploy:** Hugging Face Spaces or Render (free tier), with a live link.

---

## 9. Repo structure

```
pk-car-valuation/
├── CLAUDE.md                  # short standing conventions (generated from §2, §5)
├── SPEC.md                    # this document
├── README.md                  # framing, results, demo link, limitations, model card
├── pyproject.toml             # uv/poetry, pinned deps
├── Makefile                   # scrape / features / train / serve / test
├── config/config.yaml         # single source of truth for params
├── data/{raw,interim,processed}/   # gitignored
├── src/pkcar/
│   ├── scrape/                # two-stage, robots-aware, rate-limited
│   ├── features/              # domain FE
│   ├── data/                  # loaders, temporal-aware split
│   ├── models/                # baseline.py, boosting.py, intervals.py
│   ├── evaluation/            # metrics (PKR), SHAP, residuals
│   └── pipeline.py            # sklearn Pipeline, fit inside folds
├── api/{main.py, app.py}      # FastAPI + Streamlit
├── tests/                     # pytest
├── notebooks/                 # EDA only, never the deliverable
├── .github/workflows/ci.yml
└── Dockerfile
```

---

## 10. Deliverables

1. Public GitHub repo, clean history, one commit per phase.
2. README that reads as a model card (framing, results table, SHAP, limitations §3, demo link).
3. Live demo link (FastAPI + Streamlit).
4. A CV bullet backed by artifacts, e.g.:
   *"Built and deployed a used-car valuation service for the Pakistani market — self-scraped PakWheels data, CatBoost with conformal prediction intervals, FastAPI + Streamlit, live demo — with leakage-free CV and documented asking-vs-sale-price bias."*

> If the live demo link and repo don't exist, the CV bullet doesn't go on the CV.

---

## 11. Known limitations (state in README)

- Asking price ≠ transaction price; prices anchored upward (§3).
- Listing snapshot in time; no temporal price-trend modeling.
- Scoped to four cities; not nationally representative.
- `assembled_vs_imported` is heuristic, not authoritative.
