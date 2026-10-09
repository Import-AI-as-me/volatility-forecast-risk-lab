"""Causal risk policies and self-financing asset/cash accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_lengths(mask) -> np.ndarray:
    """True runs; false/missing days break a run."""
    a = np.asarray(mask, dtype=bool)
    changes = np.diff(np.r_[False, a, False].astype(int))
    return np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)


def residual_score(frame: pd.DataFrame, window: int = 5, lag: int = 2) -> pd.Series:
    """Only issued forecasts with outcomes observed by t-lag enter feedback."""
    if len(frame) > 1 and not frame.date.diff().iloc[1:].dt.total_seconds().eq(86400).all():
        raise ValueError("Residual feedback requires an uncompressed daily calendar")
    ratio = frame.rv / frame.forecast
    error = np.log(ratio.where((ratio > 0) & np.isfinite(ratio)))
    return error.shift(lag).rolling(window, min_periods=window).mean().clip(0, np.log(4)).fillna(0)


def rebalance_cost(pretrade: float, target: float, rate: float) -> tuple[float, float]:
    """Cost and traded notional as fractions of pre-cost NAV.

    Solve c = rate * |target*(1-c)-pretrade| exactly so the requested weight
    is a fraction of POST-cost wealth. No cost is charged to unchanged holdings.
    """
    if not (0 <= pretrade <= 1 and 0 <= target <= 1 and 0 <= rate < 1):
        raise ValueError("Long-only weights and a cost rate in [0,1) required")
    difference = target - pretrade
    notional = abs(difference) / (1 + rate * target if difference >= 0 else 1 - rate * target)
    return rate * notional, notional


def policy_path(
    frame: pd.DataFrame,
    policy: str,
    parameter: float,
    *,
    target_vol: float = 0.15,
    annualization: int = 365,
    max_weight: float = 1.0,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Daily policy with pretrade drift, overnight gaps, and proportional costs.

    frame columns: date, forecast, rv, open, close, prev_close, score.
    RV may be missing. Missing forecasts hold existing shares (no trade), not
    cash liquidation. Market boundary prices must be observed for every day.
    """
    if max_weight > 1 or max_weight <= 0:
        raise ValueError("This study supports unlevered long-only allocations")
    if policy not in {"raw", "smooth", "band", "buffer", "static"}:
        raise ValueError(f"Unknown policy {policy}")
    values = frame[["open", "close", "prev_close"]].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing/nonpositive market boundary prices; do not fabricate returns")
    out = frame.copy()
    rows = []
    closing_weight, previous_target = 0.0, 0.0
    c = cost_bps / 10_000
    for row in frame.itertuples(index=False):
        gap = row.open / row.prev_close - 1
        gap_growth = 1 + closing_weight * gap
        pretrade = np.clip(closing_weight * (1 + gap) / gap_growth, 0, 1)
        forecast = row.forecast
        if np.isfinite(forecast) and forecast > 0:
            adjusted = forecast * np.exp(parameter * row.score) if policy == "buffer" else forecast
            raw = np.clip(target_vol / np.sqrt(annualization * adjusted), 0, max_weight)
            if policy == "smooth":
                weight = parameter * raw + (1 - parameter) * previous_target
            elif policy == "band":
                weight = raw if abs(raw - pretrade) > parameter else pretrade
            elif policy == "static":
                weight = np.clip(parameter * raw, 0, max_weight)
            else:
                weight = raw
        else:
            weight = pretrade
        cost, turnover = rebalance_cost(float(pretrade), float(weight), c)
        intraday_return = row.close / row.open - 1
        intraday_growth = 1 + weight * intraday_return
        net_return = gap_growth * (1 - cost) * intraday_growth - 1
        closing_weight = weight * (1 + intraday_return) / intraday_growth
        previous_target = weight
        risk_ratio = (
            weight * np.sqrt(annualization * row.rv) / target_vol if np.isfinite(row.rv) else np.nan
        )
        rows.append(
            {
                "weight": weight,
                "pretrade_weight": pretrade,
                "turnover": turnover,
                "cost_fraction": cost,
                "net_return": net_return,
                "risk_ratio": risk_ratio,
                "risk_loss": (risk_ratio - 1) ** 2,
                "overshoot_loss": max(risk_ratio - 1, 0) ** 2
                if np.isfinite(risk_ratio)
                else np.nan,
            }
        )
    for key in rows[0] if rows else []:
        out[key] = [row[key] for row in rows]
    out["policy"], out["parameter"], out["cost_bps"] = policy, parameter, cost_bps
    return out


def path_metrics(frame: pd.DataFrame, annualization: int = 365) -> dict:
    """No annual return claims on irregularly compressed financial paths."""
    valid = frame.risk_ratio.notna()
    runs = run_lengths(frame.risk_ratio.gt(1) & valid)
    r = frame.net_return.to_numpy(float)
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    sd = np.std(r, ddof=1) if len(r) > 1 else np.nan
    return {
        "days": len(frame),
        "valid_risk_days": int(valid.sum()),
        "risk_mse": float(frame.risk_loss.mean()),
        "overshoot_mse": float(frame.overshoot_loss.mean()),
        "overshoot_fraction": float(frame.loc[valid, "risk_ratio"].gt(1).mean()),
        "overshoot_mean_run": float(runs.mean()) if len(runs) else 0.0,
        "overshoot_max_run": int(runs.max()) if len(runs) else 0,
        "mean_exposure": float(frame.weight.mean()),
        "daily_turnover": float(frame.turnover.mean()),
        "net_total_return": float(wealth[-1] - 1),
        "net_sharpe": float(np.mean(r) / sd * np.sqrt(annualization)) if sd > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
    }


def choose_policies(frames: dict[str, pd.DataFrame], config: dict) -> tuple[dict, pd.DataFrame]:
    """Choose one shared parameter per policy on validation data only."""
    validation_end = pd.Timestamp(config["validation_end"])
    validation = {s: f.loc[f.date <= validation_end].copy() for s, f in frames.items()}
    grids = {
        "raw": [1.0],
        "smooth": config["smoothing_alpha_grid"],
        "band": config["no_trade_band_grid"],
        "buffer": config["correction_gamma_grid"],
    }
    kwargs = {k: config[k] for k in ("target_vol", "annualization", "max_weight")}
    records, chosen = [], {}
    for policy, candidates in grids.items():
        scores = []
        for parameter in candidates:
            asset_scores = []
            for symbol, frame in validation.items():
                result = policy_path(frame, policy, parameter, **kwargs)
                metric = path_metrics(result, config["annualization"])
                objective = (
                    metric["risk_mse"]
                    + config["selection_turnover_penalty"] * metric["daily_turnover"]
                )
                records.append(
                    {
                        "symbol": symbol,
                        "policy": policy,
                        "parameter": parameter,
                        "objective": objective,
                        **metric,
                    }
                )
                asset_scores.append(objective)
            scores.append(float(np.mean(asset_scores)))
        chosen[policy] = float(candidates[int(np.argmin(scores))])
    # A per-asset constant scale exactly matches mean validation exposure.
    # With gamma>=0 and an unlevered cap, buffer weights never exceed raw weights
    # on available-forecast days. Bisection also accounts for held missing days.
    scales = {}
    for symbol, frame in validation.items():
        target_mean = policy_path(frame, "buffer", chosen["buffer"], **kwargs).weight.mean()
        low, high = 0.0, 1.0
        for _ in range(45):
            middle = (low + high) / 2
            trial_mean = policy_path(frame, "static", middle, **kwargs).weight.mean()
            if trial_mean < target_mean:
                low = middle
            else:
                high = middle
        scales[symbol] = (low + high) / 2
    chosen["static_scales"] = scales
    return chosen, pd.DataFrame(records)


def paired_block_interval(values, block: int = 14, repetitions: int = 2000, seed: int = 42) -> dict:
    """Circular moving-block bootstrap of daily paired differences.

    NaN calendar days remain in position. No refitting or selection is repeated;
    intervals are conditional/descriptive, not universally valid under regimes.
    """
    a = np.asarray(values, dtype=float)
    if not np.isfinite(a).any() or block < 1 or len(a) < block:
        raise ValueError("Insufficient observations for requested bootstrap")
    rng = np.random.default_rng(seed)
    count = int(np.ceil(len(a) / block))
    starts = rng.integers(0, len(a), size=(repetitions, count))
    indices = ((starts[..., None] + np.arange(block)) % len(a)).reshape(repetitions, -1)[
        :, : len(a)
    ]
    estimates = np.nanmean(a[indices], axis=1)
    return {
        "mean_difference": float(np.nanmean(a)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "block_days": block,
        "repetitions": repetitions,
        "valid_days": int(np.isfinite(a).sum()),
    }
