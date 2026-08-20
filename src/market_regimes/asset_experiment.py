"""Causal individual-asset forecasts conditioned on the global market HMM."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from market_regimes.asset_config import AssetExperimentConfig, load_asset_experiment_config
from market_regimes.assets.features import asset_feature_columns, build_asset_features
from market_regimes.assets.providers import AssetRequest, YahooChartAssetProvider
from market_regimes.assets.statistics import calculate_asset_state_statistics
from market_regimes.forecasting.asset_ledger import merge_asset_forecast_ledgers
from market_regimes.forecasting.asset_metrics import evaluate_asset_ledger
from market_regimes.forecasting.asset_online import walk_forward_asset_fold
from market_regimes.forecasting.ranking import (
    cross_sectional_daily_metrics,
    realized_return_by_predicted_rank,
    summarize_cross_sectional,
)
from market_regimes.logging_utils import configure_logging
from market_regimes.plotting.assets import create_asset_forecast_plots
from market_regimes.real_experiment import run_real_experiment

LOGGER = logging.getLogger(__name__)
MAX_LOCK_AGE_SECONDS = 6 * 60 * 60


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _exclusive_run_lock(path: Path):
    if path.exists():
        age_seconds = datetime.now(UTC).timestamp() - path.stat().st_mtime
        if age_seconds <= MAX_LOCK_AGE_SECONDS:
            raise RuntimeError(f"another asset forecast run holds {path}")
        path.unlink()
        LOGGER.warning("Removed stale asset forecast lock %s", path)
    handle = path.open("x", encoding="utf-8")
    try:
        handle.write(datetime.now(UTC).isoformat())
        handle.flush()
        yield
    finally:
        handle.close()
        path.unlink(missing_ok=True)


def _load_assets(
    config: AssetExperimentConfig,
    project_root: Path,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    cache_dir = (
        config.data.cache_dir
        if config.data.cache_dir.is_absolute()
        else project_root / config.data.cache_dir
    )
    provider = YahooChartAssetProvider(
        cache_dir, timeout_seconds=config.data.timeout_seconds
    )
    frames: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, object]] = []
    for ticker in config.data.tickers:
        frame = provider.fetch(
            AssetRequest(ticker=ticker, start=config.data.start, end=config.data.end)
        )
        source_path = Path(frame.attrs["source_path"])
        policies = sorted(str(value) for value in frame["adjustment_policy"].unique())
        manifest.append(
            {
                "ticker": ticker,
                "provider": "Yahoo Finance public chart endpoint",
                "provider_mode": frame.attrs["provider_mode"],
                "cache_path": str(source_path),
                "sha256": _sha256(source_path),
                "rows": len(frame),
                "first_observation": frame["timestamp"].min(),
                "last_observation": frame["timestamp"].max(),
                "price_used_for_returns": "price_for_returns",
                "adjustment_policies": policies,
                "adjusted_close_coverage": float(frame["adjusted_close"].notna().mean()),
                "retrieved_at": str(frame["retrieved_at"].iloc[-1]),
            }
        )
        frames[ticker] = frame
    return frames, manifest


def _load_global_context(
    real_output_dir: Path,
    real_metrics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    market = pd.read_csv(
        real_output_dir / "causal_features.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    probabilities = pd.read_csv(
        real_output_dir / "filtered_probabilities.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    mean = np.asarray(real_metrics["scaler_mean"], dtype=float)
    scale = np.asarray(real_metrics["scaler_scale"], dtype=float)
    if market.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("global market scaler metadata does not match feature matrix")
    standardized_market = pd.DataFrame(
        (market.to_numpy(dtype=float) - mean) / scale,
        index=market.index,
        columns=market.columns,
    )
    labels = {int(state): str(label) for state, label in real_metrics["state_labels"].items()}
    return standardized_market, probabilities, labels


def _latest_asset_table(
    ledger: pd.DataFrame,
    metrics: pd.DataFrame,
    state_labels: dict[int, str],
) -> pd.DataFrame:
    latest_origin = pd.Timestamp(ledger["origin_date"].max())
    latest = ledger.loc[ledger["origin_date"] == latest_origin].copy()
    latest["current_market_regime"] = np.argmax(
        latest[[f"regime_probability_{state}" for state in sorted(state_labels)]].to_numpy(),
        axis=1,
    )
    latest["current_market_regime_label"] = latest["current_market_regime"].map(
        state_labels
    )
    for state, label in state_labels.items():
        slug = label.lower().replace("-like", "").replace("-", "_")
        latest[f"probability_{slug}"] = latest[f"regime_probability_{state}"]
    latest["return_rank"] = latest.groupby("horizon_trading_days")[
        "predicted_cumulative_log_return"
    ].rank(ascending=False, method="min")
    latest["excess_return_rank"] = latest.groupby("horizon_trading_days")[
        "predicted_excess_log_return"
    ].rank(ascending=False, method="min")
    latest["direction_confidence"] = 2 * (latest["predicted_up_probability"] - 0.5).abs()
    aggregate = metrics.loc[
        metrics["scope"] == "all_folds",
        ["ticker", "horizon_trading_days", "direction_brier_score", "outperform_brier_score"],
    ]
    latest = latest.merge(
        aggregate,
        on=["ticker", "horizon_trading_days"],
        how="left",
        validate="one_to_one",
    )
    columns = [
        "origin_date",
        "forecast_for_date",
        "forecast_date_is_estimate",
        "ticker",
        "current_price",
        "current_market_regime_label",
        "probability_risk_on",
        "probability_transitional",
        "probability_risk_off",
        "horizon_trading_days",
        "predicted_cumulative_log_return",
        "predicted_up_probability",
        "predicted_excess_log_return",
        "predicted_outperform_probability",
        "asset_realized_volatility_20d",
        "asset_drawdown_252d",
        "asset_relative_strength_60d",
        "direction_confidence",
        "direction_brier_score",
        "outperform_brier_score",
        "return_rank",
        "excess_return_rank",
    ]
    return latest.loc[:, columns].sort_values(
        ["horizon_trading_days", "excess_return_rank", "ticker"], kind="stable"
    )


def _append_run_history(path: Path, record: dict[str, object]) -> None:
    history = pd.read_csv(path) if path.exists() else pd.DataFrame()
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    history.to_csv(path, index=False)


def _run_locked(
    config: AssetExperimentConfig,
    project_root: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    real_config_path = (
        config.source.real_config
        if config.source.real_config.is_absolute()
        else project_root / config.source.real_config
    )
    real_output_dir, real_metrics = run_real_experiment(real_config_path)
    market, probabilities, state_labels = _load_global_context(real_output_dir, real_metrics)
    frames, manifest = _load_assets(config, project_root)
    built = build_asset_features(
        frames,
        benchmark=config.data.benchmark,
        config=config.features,
    )
    feature_names = asset_feature_columns(config.features)
    features = built.features.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    state_statistics = calculate_asset_state_statistics(
        features,
        probabilities,
        state_labels,
        drawdown_column=f"asset_drawdown_{config.features.drawdown_window}d",
        config=config.statistics,
        random_seed=config.forecast.random_seed,
    )

    replay_parts: list[pd.DataFrame] = []
    model_bundle: dict[str, object] = {}
    initial_rows: dict[str, dict[int, int]] = {}
    input_features: tuple[str, ...] | None = None
    for fold in config.evaluation.folds:
        for ticker in config.data.tickers:
            local = (
                features.loc[features["ticker"] == ticker, ["timestamp", *feature_names]]
                .set_index("timestamp")
                .sort_index()
            )
            result = walk_forward_asset_fold(
                ticker=ticker,
                benchmark=config.data.benchmark,
                market_features=market,
                probabilities=probabilities,
                asset_features=local,
                prices=built.prices,
                fold=fold,
                config=config.forecast,
            )
            replay_parts.append(result.ledger)
            model_bundle[f"{fold.fold_id}:{ticker}"] = {
                "scaler": result.scaler,
                "models": result.models,
            }
            initial_rows[f"{fold.fold_id}:{ticker}"] = result.initial_training_rows
            if input_features is None:
                input_features = result.input_features
            elif input_features != result.input_features:
                raise RuntimeError("asset input feature order changed across assets/folds")
    replay = pd.concat(replay_parts, ignore_index=True)
    ledger_path = output_dir / "asset_forecast_ledger.csv"
    existing = (
        pd.read_csv(
            ledger_path,
            dtype={"fold_id": "string"},
            parse_dates=["origin_date", "forecast_for_date"],
        )
        if ledger_path.exists()
        else None
    )
    ledger, merge_audit = merge_asset_forecast_ledgers(existing, replay)
    metrics = evaluate_asset_ledger(ledger)
    cross_daily = cross_sectional_daily_metrics(
        ledger,
        top_k=config.evaluation.top_k,
        min_assets=config.evaluation.min_cross_sectional_assets,
    )
    cross_summary = summarize_cross_sectional(cross_daily)
    rank_summary = realized_return_by_predicted_rank(ledger)
    latest = _latest_asset_table(ledger, metrics, state_labels)

    aggregate_metrics = metrics.loc[metrics["scope"] == "all_folds"]
    validation_summary = {
        "asset_horizon_combinations": len(aggregate_metrics),
        "return_mae_beats_both_baselines": int(
            aggregate_metrics["beats_both_return_mae_baselines"].sum()
        ),
        "direction_brier_beats_both_baselines": int(
            aggregate_metrics["beats_both_direction_brier_baselines"].sum()
        ),
        "excess_mae_beats_both_baselines": int(
            aggregate_metrics["beats_both_excess_mae_baselines"].sum()
        ),
    }
    run_metrics: dict[str, Any] = {
        "status": "individual_asset_forecast_research_no_trading",
        "generated_at": datetime.now(UTC).isoformat(),
        "asset_universe": list(config.data.tickers),
        "benchmark": config.data.benchmark,
        "provider": "Yahoo Finance public chart endpoint",
        "price_for_returns": (
            "Yahoo adjusted close when available; marked raw-close fallback otherwise"
        ),
        "global_market_hmm": str(real_output_dir),
        "global_hmm_train_end": real_metrics["train_end"],
        "market_feature_count": market.shape[1],
        "regime_probability_count": probabilities.shape[1],
        "asset_feature_count": len(feature_names),
        "combined_input_dimension": len(input_features or ()),
        "asset_features": feature_names,
        "input_features": input_features,
        "targets": [
            "cumulative future adjusted-close log return",
            "positive-return direction",
            "benchmark-relative excess log return",
            "positive excess-return direction",
        ],
        "learned_models": [
            "Huber SGDRegressor for return",
            "logistic SGDClassifier for direction",
            "Huber SGDRegressor for excess return",
            "logistic SGDClassifier for outperform direction",
        ],
        "baselines": [
            "causal expanding mean and positive frequency",
            "probability-weighted HMM-regime-conditioned mean and positive frequency",
        ],
        "folds": [asdict(fold) for fold in config.evaluation.folds],
        "ledger_rows": len(ledger),
        "resolved_rows": int(ledger["resolved"].sum()),
        "pending_rows": int((~ledger["resolved"]).sum()),
        "latest_origin_date": pd.Timestamp(ledger["origin_date"].max()),
        "merge_audit": asdict(merge_audit),
        "initial_training_rows": initial_rows,
        "validation_summary": validation_summary,
        "cross_sectional_summary": cross_summary,
        "limitations": [
            "The global HMM classifications are fitted statistical labels, not objective truth.",
            (
                "Yahoo's public chart endpoint is not a guaranteed institutional "
                "data-service contract."
            ),
            (
                "Adjusted-close targets are total-return-compatible but not an "
                "executable-price series."
            ),
            "Overlapping multi-day targets create serial dependence in forecast errors.",
            "Cross-sectional top-k results are ranking diagnostics, not a portfolio backtest.",
            (
                "No costs, turnover, allocation, brokerage connection, or automated "
                "trading is included."
            ),
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "asset_features.csv", index=False)
    state_statistics.to_csv(output_dir / "asset_state_characteristics.csv", index=False)
    ledger.to_csv(ledger_path, index=False)
    latest.to_csv(output_dir / "latest_asset_forecasts.csv", index=False)
    metrics.to_csv(output_dir / "asset_forecast_metrics.csv", index=False)
    cross_daily.to_csv(output_dir / "cross_sectional_daily_metrics.csv", index=False)
    rank_summary.to_csv(output_dir / "forecast_rank_performance.csv", index=False)
    with (output_dir / "cross_sectional_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(cross_summary, handle, indent=2, default=_json_default)
    with (output_dir / "asset_data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=_json_default)
    with (output_dir / "asset_forecast_run_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metrics, handle, indent=2, default=_json_default)
    joblib.dump(
        {
            "models": model_bundle,
            "input_features": input_features,
            "state_labels": state_labels,
            "config": config,
        },
        output_dir / "asset_forecasters.joblib",
    )
    create_asset_forecast_plots(
        ledger=ledger,
        metrics=metrics,
        cross_sectional_daily=cross_daily,
        rank_summary=rank_summary,
        state_statistics=state_statistics,
        asset_features=features,
        probabilities=probabilities,
        state_labels=state_labels,
        rolling_window=config.forecast.rolling_metric_window,
        output_dir=output_dir / "plots",
    )
    _append_run_history(
        output_dir / "asset_run_history.csv",
        {
            "run_at_utc": run_metrics["generated_at"],
            "latest_origin_date": run_metrics["latest_origin_date"],
            "new_forecasts": merge_audit.new_forecasts,
            "newly_resolved": merge_audit.newly_resolved,
            "ledger_rows": len(ledger),
            "status": "success",
        },
    )
    LOGGER.info("Asset forecast outputs written to %s", output_dir)
    return output_dir, run_metrics


def run_asset_experiment(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(config_path).resolve()
    config = load_asset_experiment_config(source)
    configure_logging(config.project.log_level)
    project_root = source.parent.parent
    output_dir = (
        config.project.output_dir
        if config.project.output_dir.is_absolute()
        else project_root / config.project.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(output_dir / ".asset_run.lock"):
        return _run_locked(config, project_root, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/asset_forecast.yaml"))
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    output_dir, _ = run_asset_experiment(arguments.config)
    print(f"Individual-asset forecast diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()
