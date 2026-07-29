"""Training orchestrator (SPEC §5, §7, §8).

Grouped train/test split -> grouped CV per model (baseline LR, LightGBM, CatBoost),
every transform fit inside folds -> refit on full train -> score the held-out test set
the model never saw during CV. All metrics are raw PKR; models train on log-price.
Runs are tracked with MLflow.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import load_config, resolve_path
from .data.loaders import TARGET_COL, load_listings, log_target
from .data.split import grouped_cv, grouped_train_test_split, make_group_key
from .evaluation.metrics import format_pkr, metrics_from_log_predictions
from .models.baseline import build_baseline_model
from .models.boosting import build_catboost_model, build_lightgbm_model

logger = logging.getLogger("pkcar.train")

ModelBuilder = Callable[[dict | None], Pipeline]

MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "baseline_lr": build_baseline_model,
    "lightgbm": build_lightgbm_model,
    "catboost": build_catboost_model,
}


def cross_val_pkr(
    build_model: ModelBuilder,
    X: pd.DataFrame,
    y_price: np.ndarray,
    groups: pd.Series,
    n_splits: int,
    config: dict | None = None,
) -> dict[str, float]:
    """Grouped CV; fit a fresh pipeline inside each fold; return mean PKR metrics (SPEC §5)."""
    cv = grouped_cv(n_splits)
    y_log = log_target(y_price)
    folds: list[dict[str, float]] = []
    for k, (tr, va) in enumerate(cv.split(X, y_log, groups), start=1):
        # Fresh pipeline per fold (not sklearn.clone: CatBoost is not clone-safe with cat_features).
        model = build_model(config)
        model.fit(X.iloc[tr], y_log[tr])
        pred_log = model.predict(X.iloc[va])
        m = metrics_from_log_predictions(y_log[va], pred_log)
        folds.append(m)
        logger.info("  fold %d/%d  MAE %s  MAPE %.1f%%", k, n_splits,
                    format_pkr(m["mae"]), 100 * m["mape"])
    return {f"cv_{key}": float(np.mean([f[key] for f in folds])) for key in folds[0]}


def evaluate_holdout(
    build_model: ModelBuilder,
    X_train: pd.DataFrame,
    y_train_price: np.ndarray,
    X_test: pd.DataFrame,
    y_test_price: np.ndarray,
    config: dict | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Refit on full train, score the untouched held-out test set (SPEC §7)."""
    model = build_model(config)
    model.fit(X_train, log_target(y_train_price))
    pred_log = model.predict(X_test)
    metrics = metrics_from_log_predictions(log_target(y_test_price), pred_log)
    return model, {f"test_{k}": v for k, v in metrics.items()}


def run_training(config: dict | None = None) -> pd.DataFrame:
    """Train + evaluate all models with MLflow tracking; return a results table (SPEC §5, §7)."""
    cfg = config or load_config()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = load_listings(drop_missing_target=True, config=cfg)
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = make_group_key(df)
    n_splits = int(cfg["model"]["cv_folds"])

    tr_idx, te_idx = grouped_train_test_split(
        df, test_size=float(cfg["model"]["test_size"]),
        seed=int(cfg["project"]["random_seed"]), groups=groups,
    )
    X_train, X_test = df.iloc[tr_idx], df.iloc[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    train_groups = groups.iloc[tr_idx]
    logger.info("train %d rows / test %d rows (grouped, no shared vehicle)",
                len(tr_idx), len(te_idx))

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    rows: list[dict[str, object]] = []
    best_name, best_model, best_mae = None, None, np.inf
    for name, builder in MODEL_BUILDERS.items():
        logger.info("=== %s ===", name)
        with mlflow.start_run(run_name=name):
            cv_metrics = cross_val_pkr(builder, X_train, y_train, train_groups, n_splits, cfg)
            model, test_metrics = evaluate_holdout(
                builder, X_train, y_train, X_test, y_test, cfg
            )
            mlflow.log_param("model", name)
            mlflow.log_param("cv_strategy", cfg["model"]["cv_strategy"])
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            rows.append({"model": name, **cv_metrics, **test_metrics})
            logger.info("  CV   MAE %s  RMSE %s  MAPE %.1f%%",
                        format_pkr(cv_metrics["cv_mae"]), format_pkr(cv_metrics["cv_rmse"]),
                        100 * cv_metrics["cv_mape"])
            logger.info("  TEST MAE %s  RMSE %s  MAPE %.1f%%",
                        format_pkr(test_metrics["test_mae"]), format_pkr(test_metrics["test_rmse"]),
                        100 * test_metrics["test_mape"])
            if test_metrics["test_mae"] < best_mae:
                best_name, best_model, best_mae = name, model, test_metrics["test_mae"]

    results = pd.DataFrame(rows).sort_values("test_mae").reset_index(drop=True)
    _save_best(best_name, best_model, cfg)
    logger.info("Best model: %s (test MAE %s)", best_name, format_pkr(best_mae))
    return results


def _save_best(name: str | None, model: Pipeline | None, config: dict) -> None:
    if model is None:
        return
    import joblib
    out_dir = resolve_path(config["paths"].get("model_dir", "models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "best_model.joblib"
    joblib.dump({"name": name, "model": model}, path)
    logger.info("Saved best model -> %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PK used-car valuation models (SPEC §5).")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    results = run_training(cfg)
    print("\n=== Results (raw PKR; sorted by held-out test MAE) ===")
    with pd.option_context("display.width", 120, "display.float_format", lambda v: f"{v:,.3f}"):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
