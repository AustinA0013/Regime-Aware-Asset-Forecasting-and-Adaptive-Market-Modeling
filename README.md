# Regime-Aware Asset Forecasting and Adaptive Market Modeling

A quantitative-finance research framework for causal market-regime inference and
individual-asset forecasting. The project combines a Gaussian hidden Markov model (HMM),
forward-only state probabilities, asset-level market features, delayed-feedback online learning,
and anchored walk-forward evaluation.

> **Research status:** the software and causality tests pass, but the current asset-forecasting
> models did not outperform both causal baselines across the aggregate evaluation. This repository
> demonstrates a reproducible research process and an honest negative result; it does not present a
> validated trading strategy.

## What the code does

The project is organized as four connected experiments:

1. **Synthetic HMM recovery** generates observations from a known three-state Gaussian HMM, fits a
   new HMM, aligns its arbitrary state labels to the true states, and measures how well it recovers
   the known process.
2. **Real-market regime diagnosis** downloads daily U.S. market variables, creates causal features,
   trains the HMM on an initial period, freezes the fitted parameters, and forward-filters later
   observations.
3. **Adaptive market forecasting** predicts future market return, direction, and inferred regime.
   Each prediction is stored in a ledger and the online learner is updated only after its outcome
   becomes observable.
4. **Individual-asset forecasting** combines the global regime probabilities with asset-specific
   features for nine assets and predicts return, direction, excess return, and benchmark
   outperformance at 1-, 5-, and 21-trading-day horizons.

The current asset universe is:

```text
SPY, QQQ, IWM, AAPL, MSFT, NVDA, AMZN, GOOGL, META
```

SPY is used as the benchmark for relative-return targets.

## Why market regimes are useful

Financial returns do not always behave as if they came from one unchanging distribution. Periods
of calm growth, uncertain transition, and market stress can have different volatility,
correlation, momentum, and drawdown behavior. The HMM represents those unobserved conditions as a
small set of latent states.

The states are not directly supplied by the dataset. The model estimates them from the joint
behavior of observable market features. State numbers such as `0`, `1`, and `2` have no inherent
financial meaning; descriptive labels are assigned only after examining the fitted state
statistics.

## Core mathematics

Let `z_t` be the unobserved market state and `x_t` the feature vector observed on date `t`.
The Gaussian HMM assumes

```text
P(z_t = k | z_(t-1) = j) = A[j, k]
x_t | z_t = k ~ Normal(mu_k, Sigma_k)
```

where `A` is the state-transition matrix and each state has its own mean vector and covariance
matrix.

For causal use, the model calculates a forward-filtered probability:

```text
prediction_t(k) = sum_j alpha_(t-1)(j) * A[j, k]
alpha_t(k)      ∝ Normal(x_t; mu_k, Sigma_k) * prediction_t(k)
```

with

```text
alpha_t(k) = P(z_t = k | x_0, ..., x_t)
```

Only observations available through `t` enter `alpha_t`. There is no backward smoothing pass, so
future data cannot revise probabilities that were used as historical model inputs.

For an asset `i`, origin date `t`, and horizon `h`, the main supervised targets are

```text
future_return(i,t,h) = price(i,t+h) / price(i,t) - 1
direction(i,t,h)     = 1[future_return(i,t,h) > 0]
excess_return        = future_return(asset) - future_return(SPY)
outperformance       = 1[excess_return > 0]
```

The online learner may update from one of these targets only after date `t+h` has been reached.
This delayed-feedback rule prevents target leakage.

## Input data

### Global market variables

The real-data experiment retrieves the following public FRED series:

| Series | Financial role |
|---|---|
| `SP500` | U.S. large-cap equity price index |
| `VIXCLS` | expected near-term S&P 500 volatility |
| `DGS10` | 10-year U.S. Treasury yield |
| `DGS2` | 2-year U.S. Treasury yield |
| `BAA10Y` | Baa corporate-credit spread relative to the 10-year Treasury |

### Asset prices

The asset experiment retrieves adjusted daily OHLCV data from Yahoo Finance's public chart
endpoint. OHLCV means open, high, low, close, and trading volume. Adjusted prices account for
corporate actions when the provider supplies the required adjustment information.

Downloaded data is cached locally under `data/raw/`; it is not included in the repository.

## Features

The model creates all features with backward-looking windows. Examples include:

- daily and multi-period returns;
- realized volatility over 5, 20, and 60 trading days;
- momentum over 5, 20, 60, and 120 trading days;
- rolling drawdown from a trailing high;
- VIX level and daily change;
- 10-year minus 2-year Treasury yield-curve slope;
- Baa corporate-credit spread;
- asset return relative to SPY;
- rolling beta and correlation with SPY; and
- normalized volume and recent volume change.

The global market features and three HMM state probabilities are combined with 17 asset-level
features for each asset forecast.

## Causality safeguards

The implementation treats time ordering as part of the model specification:

- feature windows use present and past observations only;
- auxiliary market series have explicit availability lags;
- scalers and HMM parameters are fitted on training observations only;
- real-data evaluation uses frozen parameters or anchored walk-forward folds;
- HMM probabilities are produced by a forward-only recursion;
- future labels remain hidden until the forecast horizon matures;
- previously issued forecasts are immutable; and
- tests perturb future observations and verify that earlier outputs do not change.

These safeguards reduce look-ahead leakage, but they do not eliminate every real-world data issue.
The public FRED cache is latest-vintage data rather than a complete point-in-time ALFRED database.

## Repository structure

```text
configs/
  synthetic_baseline.yaml       synthetic recovery experiment
  real_data_diagnostic.yaml     frozen real-market HMM experiment
  adaptive_forecast.yaml        delayed-feedback market forecast
  asset_forecast.yaml           nine-asset walk-forward experiment

src/market_regimes/
  data/                          synthetic and FRED data providers
  assets/                        Yahoo provider, asset features, targets, statistics
  features/                      global causal market features
  models/                        HMM fitting and forward-only filtering
  forecasting/                   online learners, ledgers, metrics, rankings
  validation/                    state alignment and recovery validation
  plotting/                      deterministic diagnostic plots
  experiment.py                 synthetic experiment entry point
  real_experiment.py            real-regime experiment entry point
  adaptive_experiment.py        adaptive market forecast entry point
  asset_experiment.py           individual-asset experiment entry point

tests/                           unit, causality, ledger, and recovery tests
pyproject.toml                   package metadata and dependency ranges
requirements-lock.txt            validated environment lock file
```

## Installation

Python 3.11 or 3.12 is recommended.

```powershell
git clone https://github.com/AustinA0013/Regime-Aware-Asset-Forecasting-and-Adaptive-Market-Modeling.git
cd Regime-Aware-Asset-Forecasting-and-Adaptive-Market-Modeling
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For exact reproduction of the validated Python 3.12 environment:

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

## Run the tests

```powershell
python -m pytest -q
ruff check src tests
python -m pip check
```

The current local validation contains 39 passing tests. One Windows environment may emit a
`joblib` warning when physical CPU-core information is unavailable; `joblib` then uses the logical
core count.

## Run the experiments

```powershell
# 1. Verify HMM recovery using data whose hidden states are known
python -m market_regimes.experiment --config configs/synthetic_baseline.yaml

# 2. Fit and evaluate the frozen HMM on real market data
python -m market_regimes.real_experiment --config configs/real_data_diagnostic.yaml

# 3. Run delayed-feedback adaptive market forecasts
python -m market_regimes.adaptive_experiment --config configs/adaptive_forecast.yaml

# 4. Run individual-asset walk-forward forecasts
python -m market_regimes.asset_experiment --config configs/asset_forecast.yaml
```

Equivalent installed commands are:

```powershell
market-regimes-synthetic --config configs/synthetic_baseline.yaml
market-regimes-real --config configs/real_data_diagnostic.yaml
market-regimes-adaptive --config configs/adaptive_forecast.yaml
market-regimes-assets --config configs/asset_forecast.yaml
```

The first real-data run needs internet access. Generated datasets, fitted models, prediction
ledgers, metrics, and plots are written under `reports/artifacts/` and intentionally ignored by
Git.

## Evaluation design

The individual-asset experiment uses anchored folds:

| Fold | Training data ends | Evaluation period |
|---|---:|---:|
| 2022-2023 | 2021-12-31 | 2022-01-03 through 2023-12-29 |
| 2024-2025 | 2023-12-29 | 2024-01-02 through 2025-12-31 |
| 2026 live | 2025-12-31 | 2026-01-02 through the latest available date |

Metrics include return mean absolute error, direction Brier score, excess-return error,
benchmark-outperformance probability, cross-sectional rank correlation, Information Coefficient,
and top-k realized outcomes. Model forecasts are compared with causal expanding-history and
regime-persistence baselines.

## Current measured result

The validated nine-asset run produced 31,347 forecast records, of which 30,645 had matured at the
evaluation cutoff. Across 9 assets and 3 horizons, none of the 27 asset/horizon combinations beat
both causal baselines on all primary return, direction, and excess-return gates.

That result means:

- the data, feature, HMM, online-learning, ledger, and evaluation pipeline is operational;
- the current predictive specification has not demonstrated a reliable forecasting edge; and
- the project should not yet be extended to portfolio allocation or live trading.

Preserving this negative result is intentional. A quantitative-research portfolio should show how
claims were tested, not only display favorable backtests.

## Limitations and possible next steps

- Public data does not provide a complete historical point-in-time vintage for every variable.
- Linear online learners may not capture nonlinear regime/asset interactions.
- Hyperparameters have not undergone fully nested, multiple-testing-aware selection.
- The asset universe is small and concentrated in U.S. equities and large technology firms.
- No transaction costs, turnover constraints, position sizing, risk limits, or execution model is
  included.
- State descriptions are empirical summaries, not permanent economic identities.

Reasonable future research includes nonlinear baselines, a broader and survivorship-aware universe,
block-bootstrap uncertainty estimates, nested walk-forward model selection, point-in-time data, and
cost-aware portfolio tests after forecast gates pass.

## Disclaimer

This repository is for education, research, and portfolio demonstration. It is not investment
advice, a recommendation to trade, or evidence of expected future performance. Historical and
simulated results do not guarantee future results. Data-provider availability, revisions, corporate
actions, and implementation assumptions can materially affect the outputs.

## Copyright and use

Copyright © 2026 Austin Anderson. All rights reserved.

This repository is publicly available for viewing and portfolio/research demonstration purposes.
No license is granted to copy, modify, distribute, sublicense, or commercially use the source code
except as permitted by applicable law or GitHub's Terms of Service.
