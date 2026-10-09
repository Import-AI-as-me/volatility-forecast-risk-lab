# Study protocol, version 1.0

This file and `configs/study.json` were written before the first market-data model run.
This is an internal frozen analysis plan, **not an externally registered preregistration**.

## Question

Does clustering of volatility underprediction extend risk-target overshoot episodes?
Can a causal, residual-based risk buffer improve held-out risk tracking beyond simple
smoothing, a no-trade band, and a static reduction in exposure?

## Scope and data

BTCUSDT and ETHUSDT spot, Binance Vision monthly 5-minute archives, January 2019
through December 2025. These two surviving large cryptoassets are case studies, not
a representative asset universe. No cross-asset independence or universal efficacy claim.
UTC daily realized variance is the sum of valid consecutive 5-minute squared log returns.
Incomplete days are explicitly marked, not interpolated.

## Time and execution

Initial estimation: 2019–2021. Validation: 2022–2023. Held-out test: 2024–2025.
Monthly expanding model refits may use newly observed historical test targets; this
is a fixed walk-forward rule, not hyperparameter tuning on test performance.
At the opening of day t, predictors and residual feedback use observations through
day t−2. The full intervening day is an intentionally conservative processing gap.
The target is day t variance: effectively a two-calendar-day-ahead forecast.
No use of the just-finished close to assume execution at that same close.

## Models and controls

Persistence, EWMA (span 10), log-HAR linear regression, and a small histogram
gradient boosting regressor using exactly the HAR predictors. Log models use a
training-only smearing factor. Architecture and model hyperparameters are fixed.
logHAR is the primary decision model; no choice of winning forecast model after testing.

## Decision experiment

Each asset is a separate long-only, unlevered asset-plus-zero-interest-cash account.
Target annual volatility is 15%, with 365-day annualization. A daily target weight
is clipped to [0,1]. Simple returns are open-to-close. Portfolio positions drift
between rebalances; turnover is measured against pretrade, return-drifted weights.
The bar open is an execution reference, not a claim of a fill at that exact price.
Daily open/previous-close jumps and costs are accounted for by the backtest.
Proportional costs are assumed scenarios (0/5/10/20 bps of one-way notional), not
measured spreads, slippage or capacity. Cash earns zero; funding and leverage are absent.

Policies: raw targeting; smoothed targeting; no-trade band; a residual buffer; and
a static raw-target multiplier matched to the buffer's mean validation exposure.
The residual buffer uses the nonnegative part of the trailing 5-day mean log
(realized variance / issued forecast), delayed two calendar days, clipped at log(4).
It multiplies the forecast variance by exp(gamma × this score).
Gamma=0 is included so selecting no correction is an admissible result.
Each policy's parameters are selected by mean symmetric squared relative risk
deviation plus 0.1 times daily one-way turnover, on validation only, averaged
equally over assets. Symmetric risk tracking penalizes simply moving to cash.
Selection and static matching use zero-cost validation paths. The static exposure
match is estimated on validation and frozen, not retuned on test.

## Endpoints and uncertainty

The realized risk diagnostic is weight × sqrt(365 × daily RV), a one-day
ex-post proxy, not a conditional volatility estimate or a calibrated probability.
It does not reconstruct intraday portfolio weight drift or old-position opening-gap
risk; those are distinct from the self-financing financial return calculation.
Primary endpoint: mean squared (realized risk / target − 1), residual buffer
minus static exposure-matched raw policy. Overshoot frequency, mean and maximum
run duration, overshoot severity, average exposure and turnover are reported too.
Forecast metrics: QLIKE = RV/f − log(RV/f) − 1 and squared log error.
Net performance and drawdown are secondary scenario diagnostics, not proof of alpha.

Paired moving-block bootstrap intervals use common dates and preserve cross-asset
dependence; block lengths 7,14,28, 2000 draws, fixed seed. They are descriptive,
conditional on fitted models and this selected sample, not causal inference or
protection from nonstationarity/multiple comparisons. Per-asset results stay visible.

## Controlled experiment and stopping rule

At constant true variance with a memoryless allocation, permuting an identical
multiset of errors cannot change any average pointwise loss. It can change run
duration and turnover. The simulation must demonstrate this distinction, not
manufacture an average-risk effect of persistence.

Report the chosen gamma even when zero. Keep negative/mixed results. Do not alter
the test interval, metric, parameters or assets to make the correction win.
Implementation fixes are allowed and documented; scientific changes need a new
protocol version and a separately labelled exploratory analysis.

## Implementation clarifications

After initial evaluation, wording was clarified to explicitly describe the existing
zero-cost parameter selection and the existing risk-proxy approximation. No
parameter, split, endpoint, asset, or selection rule was changed.
