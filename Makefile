# Task shortcuts (SPEC §9). All params come from config/config.yaml (SPEC §8).
.PHONY: install lint test scrape train interpret serve-model api demo docker clean

install:          ## install deps (incl. dev group) via uv
	uv sync --dev

lint:             ## ruff lint
	uv run ruff check .

test:             ## pytest
	uv run pytest -q

scrape:           ## two-stage PakWheels scrape (SPEC §2): Stage 1 list then Stage 2 detail
	uv run python -m pkcar.scrape stage1
	uv run python -m pkcar.scrape stage2

train:            ## grouped-CV train baseline + LightGBM + CatBoost, log to MLflow (SPEC §5)
	uv run python -m pkcar.train

interpret:        ## conformal intervals + SHAP + residual diagnostics (SPEC §6–§7)
	uv run python -m pkcar.interpret

serve-model:      ## build the served conformal artifact models/serving_model.joblib (SPEC §8)
	uv run python -m pkcar.serving build

api:              ## FastAPI /predict — point + interval (SPEC §8)
	uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

demo:             ## Streamlit demo, loads model in-process (SPEC §8)
	uv run streamlit run api/app.py

docker:           ## build the deploy image (run serve-model first)
	docker build -t pkcar .

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
