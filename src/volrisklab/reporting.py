"""Publication-style figures and a results memo generated from computed outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "raw": "#64748b",
    "smooth": "#b45309",
    "band": "#7c3aed",
    "buffer": "#007f82",
    "static": "#be123c",
}
LABELS = {
    "raw": "Raw HAR",
    "smooth": "Smoothed HAR",
    "band": "No-trade band",
    "buffer": "Residual buffer",
    "static": "Static exposure match",
}


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}" if np.isfinite(value) else "NA"
        return str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *rows])


def make_figures(daily, forecasts, metrics, paths, output_dir: Path, config: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.16,
            "savefig.dpi": 180,
        }
    )
    test_start = pd.Timestamp(config["test_start"])
    symbols = list(config["symbols"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    for ax, symbol in zip(axes, symbols):
        g = daily.loc[(daily.symbol == symbol) & (daily.date >= test_start)]
        p = forecasts.loc[
            (forecasts.symbol == symbol)
            & forecasts.model.eq("logHAR")
            & (forecasts.date >= test_start)
        ]
        ax.plot(
            g.date,
            np.sqrt(365 * g.rv),
            color="#94a3b8",
            linewidth=0.8,
            label="Realized variance proxy → annualized volatility",
        )
        ax.plot(
            p.date,
            np.sqrt(365 * p.forecast),
            color=COLORS["buffer"],
            linewidth=1.0,
            label="Causal HAR forecast (lag 2)",
        )
        ax.set(title=symbol, ylabel="Annualized volatility", yscale="log")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Held-out 2024–2025 · observed risk and causal forecasts", fontsize=14)
    fig.savefig(output_dir / "forecast_paths.png")
    plt.close(fig)

    fm = pd.read_csv(output_dir / "forecast_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    order = ["persistence", "EWMA10", "logHAR", "ML_HAR"]
    for ax, symbol in zip(axes, symbols):
        g = fm.loc[fm.split.eq("test") & fm.symbol.eq(symbol)].set_index("model").loc[order]
        ax.bar(order, g.qlike, color=["#94a3b8", "#64748b", "#007f82", "#b45309"], width=0.65)
        ax.set(
            title=f"{symbol} · {int(g.valid_days.iloc[0])} common days",
            ylabel="Mean QLIKE (lower is better)",
        )
        ax.tick_params(axis="x", labelsize=9)
    fig.suptitle("Forecast quality · identical test dates within each asset", fontsize=13)
    fig.savefig(output_dir / "forecast_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    test_metrics = metrics.loc[metrics.split.eq("test") & metrics.cost_bps.eq(10)]
    for ax, symbol in zip(axes, symbols):
        g = test_metrics.loc[test_metrics.symbol.eq(symbol)]
        for row in g.itertuples():
            ax.scatter(
                100 * row.mean_exposure,
                row.risk_mse,
                color=COLORS[row.policy],
                marker={"raw": "o", "smooth": "s", "band": "^", "buffer": "D", "static": "x"}[
                    row.policy
                ],
                s=80,
                label=LABELS[row.policy],
                linewidths=1.5,
                alpha=0.85,
            )
        ax.set(title=symbol, xlabel="Mean asset exposure (%)", ylabel="Symmetric relative risk MSE")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Risk control and exposure · 2024–2025", fontsize=14)
    fig.savefig(output_dir / "risk_exposure.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    test = paths.loc[(paths.date >= test_start) & paths.cost_bps.eq(10)]
    for ax, symbol in zip(axes, symbols):
        for policy in ["raw", "smooth", "band", "buffer", "static"]:
            g = test.loc[test.symbol.eq(symbol) & test.policy.eq(policy)].sort_values("date")
            ax.plot(
                g.date,
                (1 + g.net_return).cumprod(),
                color=COLORS[policy],
                label=LABELS[policy],
                linewidth=1.4,
                linestyle="--" if policy == "static" else "-",
            )
        ax.set(title=symbol, ylabel="Wealth / initial wealth")
        ax.tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Secondary diagnostic · assumed 10 bps one-way costs, zero-interest cash", fontsize=12
    )
    fig.savefig(output_dir / "wealth_scenarios.png")
    plt.close(fig)


def write_report(fm, metrics, intervals, chosen, output_dir: Path, config: dict) -> None:
    audit = json.loads((output_dir / "data_audit.json").read_text())
    data_table = pd.DataFrame(audit["symbols"])[
        ["symbol", "raw_rows", "days", "valid_rv_days", "missing_bars"]
    ]
    test_fm = fm.loc[
        fm.split.eq("test"),
        [
            "symbol",
            "model",
            "valid_days",
            "qlike",
            "log_mse",
            "log_error_acf1",
            "underprediction_max_run",
        ],
    ]
    dm = metrics.loc[metrics.split.eq("test") & metrics.cost_bps.eq(10)].copy()
    risk_table = dm[
        [
            "symbol",
            "policy",
            "risk_mse",
            "overshoot_fraction",
            "overshoot_mean_run",
            "overshoot_max_run",
            "mean_exposure",
            "daily_turnover",
        ]
    ]
    perf = dm[["symbol", "policy", "net_total_return", "net_sharpe", "max_drawdown"]]
    primary = intervals.loc[intervals.primary.eq(True)]
    p14 = primary.loc[primary.block_days.eq(14)].iloc[0]
    if chosen["buffer"] == 0:
        verdict = "Validation selected gamma = 0: the proposed residual correction was not retained. The data do not support a benefit of this correction under the frozen selection rule."
    elif p14.ci_high < 0:
        verdict = "The buffer reduced the primary held-out risk loss relative to the validation exposure-matched static control; the 14-day descriptive bootstrap interval lies below zero. This is sample-specific evidence, not causal identification or a profitability claim."
    elif p14.ci_low > 0:
        verdict = "The buffer increased the primary held-out risk loss relative to the validation exposure-matched static control; the 14-day descriptive bootstrap interval lies above zero. This is a negative result for the proposed correction."
    else:
        verdict = "The 14-day descriptive bootstrap interval for the primary comparison crosses zero. The experiment does not establish an improvement over the validation exposure-matched static control."
    sim = pd.read_csv(output_dir / "simulation_summary.csv")[
        ["scenario", "mean_qlike", "risk_tracking_mse", "longest_overshoot_run", "total_turnover"]
    ]
    content = f"""# VolRiskLab — research results

## Finding

{verdict}

The prespecified primary contrast is **buffer minus static exposure-matched control**,
mean squared relative risk deviation, averaged on common days over both assets.
It uses zero-cost accounting paths; the displayed scenario tables use 10 bps.
Negative is better. Estimate: **{p14.mean_difference:.6f}**;
14-day block-bootstrap 95% interval: **[{p14.ci_low:.6f}, {p14.ci_high:.6f}]**.
All intervals and other comparisons are retained in `risk_comparisons.csv`.

This is an executed, reproducible research project. It is not a submitted paper,
a demonstrated trading edge, or a claim of a novel forecasting architecture.

## Question and design

Can temporally clustered volatility underprediction extend risk-target overshoot?
Does a feedback rule based only on previously observed residuals improve decisions
beyond a static reduction in exposure, simple smoothing, and a no-trade band?

The [protocol](../../docs/protocol.md) and [configuration](../../configs/study.json)
were fixed before market-model evaluation. They were not externally preregistered.
Two surviving large cryptoassets are deliberately small case studies.
2019–2021 initializes estimation, 2022–2023 selects policy parameters, and
2024–2025 is the historical held-out evaluation. Expanding monthly refits use
newly observed past targets in test; hyperparameters are not retuned on test.

## Data and audit

{markdown_table(data_table)}

Source: [Binance Vision](https://github.com/binance/binance-public-data), spot
5-minute OHLCV archives, UTC, January 2019–December 2025. All 168 archives are
verified against their vendor SHA256 sidecars. `data_manifest.json` preserves
URLs, hashes and retrieval times. `data_audit.json` lists missing intervals.
2025 microsecond timestamps and earlier millisecond timestamps are normalized.
No missing bars are interpolated and no return bridges a missing interval.
The first dataset day lacks a prior closing observation and has no valid RV.
The two assets share venue outage dates; they are not independent replications.

RV is the sum of 288 consecutive squared 5-minute log returns, attributed to the
UTC day of the ending bar. Incomplete days retain missing RV. They remain on
the calendar. Forecast and risk metrics exclude invalid targets; financial
accounting still keeps days with observed exact opening/closing boundaries.
Missing forecasts trigger a hold of existing shares, not a fictitious liquidation.

## Forecasts: the actual machine learning component

Persistence and EWMA(span 10) provide low-cost controls. Log-HAR uses linear
regression on log daily, five-day and 22-day mean RV. ML_HAR uses a small histogram
gradient boosting model with **the identical three features**. Its purpose is
to test a nonlinear mapping, not a richer information set. Log predictions use
training-residual smearing to return to the variance scale.

At the opening of held day t, all predictors and feedback stop at t−2. This leaves
one full processing day and is deliberately slower than an ideal production
pipeline. It is a two-calendar-day-ahead target, not a conventional lag-one HAR
benchmark. Models refit monthly; archived issued forecasts generate feedback.
Training-only smearing is not a guarantee of out-of-sample calibration.

QLIKE = RV/f − log(RV/f) − 1. Lower is better. All four models use common valid
target dates within each asset for this table; the number of observations is
daily, not the number of five-minute bars.

{markdown_table(test_fm)}

![Forecast scores](forecast_comparison.png)

![Forecast paths](forecast_paths.png)

## Controlled mechanism experiment

Under constant true variance and memoryless volatility targeting, rearranging an
identical multiset of forecast errors leaves every mean pointwise loss unchanged.
It changes the sequence of weights, overshoot run lengths, and turnover. Therefore
"persistent errors are worse" needs an explicit path-dependent loss definition.
This elementary invariant is a diagnostic, not a claimed new theorem.

{markdown_table(sim)}

The simulation's `risk_tracking_mse` is squared **absolute annualized volatility** error;
the empirical `risk_mse` below is squared **relative** error. Their scales differ
by target-volatility squared. See [the derivation](../../docs/theory.md).

![Controlled error ordering](simulation.png)

## Policies and validation selection

The primary decision model was fixed to logHAR before testing. Asset and cash are
unlevered; target annual volatility is 15%, using 365-day annualization. Cash earns
zero. The residual buffer multiplies forecast variance by exp(gamma × score),
where score is the positive part of a trailing five-day mean log(RV/issued forecast),
clipped at log(4), and delayed two days. A nonzero score captures recent positive
bias; it does not uniquely identify the persistence mechanism.

Selected parameters: `{json.dumps(chosen, sort_keys=True)}`.

Selection minimizes validation symmetric relative risk MSE + 0.1 × daily turnover,
equally averaged across assets. The symmetry penalizes avoiding risk by sitting
entirely in cash. Selection and static exposure matching use zero-cost validation
paths. The static multiplier matches the buffer's mean validation exposure for
each asset and is then frozen. Test exposure is reported, not forcibly matched.
All candidate trials are in `validation_trials.csv`, including gamma = 0.

## Held-out risk and exposure

{markdown_table(risk_table)}

The daily proxy is opening weight × sqrt(365 × RV), divided by the 15% target.
This is an ex-post risk diagnostic, not conditional volatility or a risk probability.
It does not reconstruct within-day portfolio-weight drift or the previous position's
opening-gap risk. Missing-risk days break run counts, potentially shortening them.
The financial return calculation separately includes overnight gaps and holding drift.

![Risk and exposure](risk_exposure.png)

Primary comparison, with alternative block lengths:

{markdown_table(primary[["block_days", "valid_days", "mean_difference", "ci_low", "ci_high"]], 6)}

Paired circular moving-block resampling uses the same dates across assets and
keeps missing calendar positions. These percentile intervals are conditional on
the fitted forecasts, selected rules, and historical sample. They do not repeat
estimation or selection and are not distribution-free guarantees under regime
changes. Other contrasts are exploratory; no multiple-comparison correction.

## Secondary financial diagnostics

Assumed cost: 10 bps per one-way traded notional, with exact self-financing cost
accounting and pretrade weights drifted by asset returns. The first purchase is
charged. Terminal wealth marks remaining holdings to market without liquidation.
The daily bar open is a reference price, not verified execution. OHLCV cannot
identify actual spreads, market impact, queue fills, or strategy capacity.

{markdown_table(perf)}

`net_total_return` covers the two-year test period, not one year. Sharpe uses zero
cash interest and sqrt(365). Cost scenarios 0/5/10/20 bps are all preserved in
`decision_metrics.csv`; costs are assumptions, not observed transaction expenses.

![Scenario wealth](wealth_scenarios.png)

## Interpretation and limits

1. Temporal ordering can alter duration and turnover without changing average
   error. The controlled experiment establishes only this limited mechanism.
2. The empirical correction also changes forecast level and exposure. The static
   control helps diagnose that confound but does not exactly equalize test exposure.
3. This is risk allocation, not return prediction or evidence of alpha. A higher
   historical Sharpe alone is not the study's success criterion.
4. Only two selected assets, one venue and two final test years are evaluated.
   Archived data can be revised; saved hashes identify this snapshot, not a
   point-in-time database. Five-minute RV is a noisy proxy for latent variance.
5. The learned model is intentionally small. Millions of bars produce only
   thousands of daily labels; five-minute rows are not independent daily samples.
6. Novelty and a publishable contribution have not been established. See
   [related work](../../docs/references.md) for directly overlapping literature.

## Reproduction and licensing

From the repository root: `volrisklab data`, then `volrisklab run`.
An offline mechanism demo is `volrisklab simulate`. Run `pytest` for timing,
missing-data, accounting and invariant checks. Exact environment and source/data
hashes are recorded in `run_provenance.json` and `requirements-lock.txt` at repo root.
Raw data and full per-day paths are excluded from the publication bundle; commands
regenerate them. Code uses the MIT license; market-derived outputs in this directory
are attributed to Binance Vision and supplied under CC BY-NC-SA 4.0 as documented
in [DATA_LICENSE](../../DATA_LICENSE.md). This is noncommercial research.
"""
    (output_dir / "RESULTS.md").write_text(content, encoding="utf-8")
