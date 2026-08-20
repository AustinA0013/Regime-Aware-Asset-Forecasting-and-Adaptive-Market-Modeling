"""Configuration-driven universe and anchored-fold tests."""

from market_regimes.asset_config import load_asset_experiment_config


def test_asset_configuration_loads_full_example_universe() -> None:
    config = load_asset_experiment_config("configs/asset_forecast.yaml")
    assert config.data.benchmark == "SPY"
    assert len(config.data.tickers) == 9
    assert config.forecast.horizons == (1, 5, 21)
    assert len(config.evaluation.folds) == 3
