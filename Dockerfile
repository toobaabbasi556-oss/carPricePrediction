# PK Used-Car Valuation — deploy image (SPEC §8).
# Single container serving the Streamlit demo, which loads the conformal model in-process.
# Build the serving artifact locally FIRST (needs the local raw data, which is gitignored):
#   python -m pkcar.serving build      # writes models/serving_model.joblib
#   docker build -t pkcar .
#   docker run -p 8501:8501 pkcar
# For HF Spaces / Render, the platform sets $PORT; we bind to it.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Install dependencies first (cached layer) — runtime deps only, no dev tooling.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project code, config, and the prebuilt serving artifact.
COPY src ./src
COPY api ./api
COPY config ./config
COPY models/serving_model.joblib ./models/serving_model.joblib
RUN uv sync --frozen --no-dev

EXPOSE 8501

# $PORT is provided by HF Spaces / Render; default to 8501 locally.
CMD ["sh", "-c", "uv run streamlit run api/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
