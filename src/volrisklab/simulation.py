"""A controlled diagnostic of forecast-error ordering, without market data.

Every scenario has the same true variance and exactly the same forecast errors.
Only their temporal order changes. This isolates runs and turnover from marginal
forecast accuracy and pointwise risk-tracking loss.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

ANNUALIZATION = 365
TRUE_DAILY_VARIANCE = 0.04 / ANNUALIZATION
TARGET_ANNUAL_VOL = 0.15
MAX_WEIGHT = 1.0
INITIAL_WEIGHT = 0.0
DEFAULT_STEPS = 240
DEFAULT_ERROR_AMPLITUDE = 0.4
BLOCK_LENGTHS = (1, 5, 20)


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return lengths of maximal true runs, counting runs at both endpoints."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def compute_scenarios(
    n_steps: int = DEFAULT_STEPS,
    error_amplitude: float = DEFAULT_ERROR_AMPLITUDE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return path rows and summary rows for three permutations of +/- errors.

    ``n_steps`` must contain complete cycles for every block length, so all
    scenarios have the same counts, starting sign, and ending sign. The defaults
    leave the weight cap inactive, making the two-weight calculation transparent.
    No stochastic returns are drawn: annualized risk is a model-based exposure
    proxy, not an empirical rolling realized-volatility estimate.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if any(n_steps % (2 * length) for length in BLOCK_LENGTHS):
        raise ValueError("n_steps must be divisible by 40 for complete block cycles")
    if not np.isfinite(error_amplitude) or error_amplitude <= 0:
        raise ValueError("error_amplitude must be finite and positive")
    # Avoid overflow in the deliberately bounded two-point diagnostic.
    if error_amplitude > 10:
        raise ValueError("error_amplitude must not exceed 10")

    target_daily_vol = TARGET_ANNUAL_VOL / np.sqrt(ANNUALIZATION)
    paths: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for block_length in BLOCK_LENGTHS:
        scenario = "alternating" if block_length == 1 else f"blocks_{block_length}"
        cycle = np.repeat([-error_amplitude, error_amplitude], block_length)
        errors = np.tile(cycle, n_steps // len(cycle))
        forecast = TRUE_DAILY_VARIANCE * np.exp(errors)
        weights = np.minimum(MAX_WEIGHT, target_daily_vol / np.sqrt(forecast))
        annualized_risk = np.sqrt(ANNUALIZATION * TRUE_DAILY_VARIANCE) * weights
        overshoot = annualized_risk > TARGET_ANNUAL_VOL
        turnover = np.abs(np.diff(np.concatenate(([INITIAL_WEIGHT], weights))))
        cumulative_turnover = np.cumsum(turnover)
        # QLIKE(y, y_hat) = y/y_hat - log(y/y_hat) - 1.
        qlike = np.expm1(-errors) + errors
        squared_log_error = errors**2
        tracking_squared_error = (annualized_risk - TARGET_ANNUAL_VOL) ** 2
        runs = _run_lengths(overshoot)
        switches = int(np.count_nonzero(np.diff(errors)))
        weight_negative_error = float(weights[0])
        weight_positive_error = float(weights[block_length])
        turnover_formula = abs(weight_negative_error - INITIAL_WEIGHT) + switches * abs(
            weight_negative_error - weight_positive_error
        )

        for step in range(n_steps):
            paths.append(
                {
                    "scenario": scenario,
                    "step": step,
                    "block_length": block_length,
                    "true_daily_variance": TRUE_DAILY_VARIANCE,
                    "log_variance_error": float(errors[step]),
                    "forecast_daily_variance": float(forecast[step]),
                    "weight": float(weights[step]),
                    "annualized_risk_proxy": float(annualized_risk[step]),
                    "target_annual_vol": TARGET_ANNUAL_VOL,
                    "risk_overshoot": bool(overshoot[step]),
                    "qlike": float(qlike[step]),
                    "squared_log_variance_error": float(squared_log_error[step]),
                    "risk_tracking_squared_error": float(tracking_squared_error[step]),
                    "turnover": float(turnover[step]),
                    "cumulative_turnover": float(cumulative_turnover[step]),
                }
            )

        summaries.append(
            {
                "scenario": scenario,
                "n_steps": n_steps,
                "block_length": block_length,
                "error_amplitude": error_amplitude,
                "mean_qlike": float(np.mean(qlike)),
                "mean_squared_log_variance_error": float(np.mean(squared_log_error)),
                "risk_tracking_mse": float(np.mean(tracking_squared_error)),
                "overshoot_fraction": float(np.mean(overshoot)),
                "overshoot_run_count": len(runs),
                "longest_overshoot_run": int(np.max(runs)) if len(runs) else 0,
                "mean_overshoot_run": float(np.mean(runs)) if len(runs) else 0.0,
                "weight_at_negative_error": weight_negative_error,
                "weight_at_positive_error": weight_positive_error,
                "number_of_weight_switches": switches,
                "initial_weight": INITIAL_WEIGHT,
                "initial_turnover": float(turnover[0]),
                "total_turnover": float(np.sum(turnover)),
                "turnover_from_switch_formula": float(turnover_formula),
            }
        )

    return paths, summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_simulation(
    paths: list[dict[str, Any]], summaries: list[dict[str, Any]], destination: Path
) -> None:
    # Use an explicit noninteractive canvas so this also works in headless CI.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 7.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    colors = ("#2463a6", "#b4651b", "#407652")
    risk_values = [row["annualized_risk_proxy"] for row in paths]
    risk_min, risk_max = min(risk_values), max(risk_values)
    padding = max(0.01, 0.15 * (risk_max - risk_min))

    for axis, summary, color in zip(axes[:3], summaries, colors):
        scenario = summary["scenario"]
        rows = [row for row in paths if row["scenario"] == scenario]
        steps = np.array([row["step"] for row in rows])
        risk = np.array([row["annualized_risk_proxy"] for row in rows])
        unit = "observation" if summary["block_length"] == 1 else "observations"
        label = f"Error blocks of {summary['block_length']} {unit}"
        axis.step(steps, risk, where="post", color=color, linewidth=1.3)
        axis.axhline(TARGET_ANNUAL_VOL, color="#505050", linestyle="--", linewidth=1)
        axis.fill_between(
            steps,
            TARGET_ANNUAL_VOL,
            risk,
            where=risk > TARGET_ANNUAL_VOL,
            step="post",
            color=color,
            alpha=0.16,
        )
        axis.set_title(
            f"{label}\nLongest overshoot: {summary['longest_overshoot_run']} {unit}",
            fontsize=10,
        )
        axis.set(xlabel="Observation", ylabel="Annualized risk proxy")
        axis.set_ylim(risk_min - padding, risk_max + padding)
        axis.grid(alpha=0.2)
        axes[3].plot(
            steps,
            [row["cumulative_turnover"] for row in rows],
            color=color,
            label=f"Blocks {summary['block_length']}",
        )

    axes[3].set(
        title="Ordering changes turnover, including initial entry",
        xlabel="Observation",
        ylabel="Cumulative absolute weight change",
    )
    axes[3].legend(frameon=False)
    axes[3].grid(alpha=0.2)
    figure.suptitle(
        "Same forecast-error distribution and tracking loss; different run durations\n"
        "Constant true variance; memoryless capped volatility targeting; no market returns",
        fontsize=12,
    )
    figure.savefig(destination, dpi=160, facecolor="white")


def run_simulation(output_dir: Path) -> dict[str, Any]:
    """Write the reproducible diagnostic and return JSON-serializable metadata.

    Outputs are ``simulation_paths.csv``, ``simulation_summary.csv``, and
    ``simulation.png``. Turnover always includes entering the first position from
    zero; it excludes a terminal liquidation and is not multiplied by a fee rate.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, summaries = compute_scenarios()
    paths_csv = output_dir / "simulation_paths.csv"
    summary_csv = output_dir / "simulation_summary.csv"
    figure_path = output_dir / "simulation.png"
    _write_csv(paths_csv, paths)
    _write_csv(summary_csv, summaries)
    _plot_simulation(paths, summaries, figure_path)
    return {
        "paths_csv": str(paths_csv),
        "summary_csv": str(summary_csv),
        "figure": str(figure_path),
        "parameters": {
            "n_steps": DEFAULT_STEPS,
            "error_amplitude": DEFAULT_ERROR_AMPLITUDE,
            "annualization": ANNUALIZATION,
            "true_daily_variance": TRUE_DAILY_VARIANCE,
            "target_annual_vol": TARGET_ANNUAL_VOL,
            "max_weight": MAX_WEIGHT,
            "initial_weight": INITIAL_WEIGHT,
            "initial_turnover_included": True,
            "terminal_liquidation_included": False,
        },
        "summary": summaries,
    }
