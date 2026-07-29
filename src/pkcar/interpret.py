"""Uncertainty + interpretability report (SPEC §6, §7).

Fits conformal prediction intervals on the CatBoost model (grouped train/calibration),
scores empirical coverage + mean width on the untouched held-out test set, and writes
SHAP (summary + dependence) and residual-diagnostic figures. Tracked with MLflow.
"""

from __future__ import annotations

import argparse
import logging

import mlflow
import pandas as pd

from .config import load_config, resolve_path
from .data.loaders import TARGET_COL, load_listings
from .data.split import grouped_train_test_split, make_group_key
from .evaluation.metrics import format_pkr, interval_metrics_pkr
from .evaluation.residuals import error_by_band, error_by_make, save_residuals_vs_predicted
from .evaluation.shap_plots import (
    compute_shap_values,
    mean_abs_shap,
    save_shap_dependence,
    save_shap_summary,
)
from .models.boosting import build_catboost_model
from .models.intervals import ConformalIntervalModel

logger = logging.getLogger("pkcar.interpret")

DEPENDENCE_FEATURES = ("age", "mileage_km", "engine_cc")


def run_interpret(config: dict | None = None) -> dict[str, float]:
    """Conformal intervals + SHAP + residual diagnostics on the held-out test (SPEC §6, §7)."""
    cfg = config or load_config()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fig_dir = resolve_path("reports/figures")

    df = load_listings(drop_missing_target=True, config=cfg)
    y = df[TARGET_COL].to_numpy(dtype=float)
    groups = make_group_key(df)
    tr, te = grouped_train_test_split(
        df, test_size=float(cfg["model"]["test_size"]),
        seed=int(cfg["project"]["random_seed"]), groups=groups,
    )
    X_train, X_test = df.iloc[tr], df.iloc[te]
    y_train, y_test = y[tr], y[te]

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="catboost_conformal_interpret"):
        # --- Conformal intervals (SPEC §6) ---
        conf = ConformalIntervalModel(build_catboost_model, config=cfg)
        conf.fit(X_train, y_train, groups.iloc[tr])
        point, lower, upper = conf.predict(X_test)
        im = interval_metrics_pkr(y_test, lower, upper)
        target_cov = 1 - conf.alpha
        logger.info("Conformal %.0f%% interval: coverage %.1f%%  mean width %s",
                    100 * target_cov, 100 * im["coverage"], format_pkr(im["mean_width"]))
        mlflow.log_metrics({
            "interval_coverage": im["coverage"],
            "interval_mean_width": im["mean_width"],
            "interval_target": target_cov,
        })

        # --- SHAP (SPEC §6) ---
        shap_values, xt, _ = compute_shap_values(conf.model_, X_test)
        summary_path = save_shap_summary(shap_values, xt, fig_dir / "shap_summary.png")
        logger.info("SHAP top drivers:\n%s", mean_abs_shap(shap_values, xt).head(8).round(3))
        for feat in DEPENDENCE_FEATURES:
            save_shap_dependence(shap_values, xt, feat, fig_dir / f"shap_dependence_{feat}.png")
        mlflow.log_artifact(str(summary_path))

        # --- Residual diagnostics (SPEC §7) ---
        resid_path = save_residuals_vs_predicted(y_test, point, fig_dir / "residuals_vs_pred.png")
        band = error_by_band(y_test, point)
        make = error_by_make(X_test["make"], y_test, point)
        logger.info("Error by price band:\n%s", band.round(3))
        logger.info("Error by make (worst MAPE):\n%s", make.round(3))
        mlflow.log_artifact(str(resid_path))

    return {"coverage": im["coverage"], "mean_width": im["mean_width"], "target": target_cov}


def main() -> None:
    parser = argparse.ArgumentParser(description="Interval + SHAP + residual report (SPEC §6, §7).")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    with pd.option_context("display.width", 120):
        run_interpret(cfg)


if __name__ == "__main__":
    main()
