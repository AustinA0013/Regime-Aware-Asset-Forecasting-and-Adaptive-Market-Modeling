"""Publication-oriented diagnostic plots for regime research."""

from market_regimes.plotting.forecast import create_adaptive_forecast_plots
from market_regimes.plotting.real import create_real_diagnostic_plots
from market_regimes.plotting.synthetic import create_synthetic_diagnostic_plots

__all__ = [
    "create_adaptive_forecast_plots",
    "create_real_diagnostic_plots",
    "create_synthetic_diagnostic_plots",
]
