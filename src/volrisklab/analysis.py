"""Fixed-protocol empirical experiment and transparent machine-readable outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .decisions import (
    choose_policies,
    paired_block_interval,
    path_metrics,
    policy_path,
    residual_score,
    run_lengths,
)
from .forecast import make_forecasts
from .simulation import run_simulation


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecast_metrics(predictions: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    # Same target dates for every model within each asset.
    common = (
        predictions.assign(valid=lambda d: d.forecast.gt(0) & d.rv.gt(0))
        .groupby(["date", "symbol"])
        .valid.all()
    )
    data = predictions.join(common.rename("common"), on=["date", "symbol"])
    for split, mask in [
        ("validation", data.date <= pd.Timestamp(config["validation_end"])),
        ("test", data.date >= pd.Timestamp(config["test_start"])),
    ]:
        for (symbol, model), g in data.loc[mask].groupby(["symbol", "model"]):
            ratio = (g.rv / g.forecast).where(g.common)
            logerror = np.log(ratio)
            runs = run_lengths(ratio.gt(1))
            rows.append(
                {
                    "split": split,
                    "symbol": symbol,
                    "model": model,
                    "valid_days": int(ratio.notna().sum()),
                    "calendar_days": len(g),
                    "qlike": float((ratio - logerror - 1).mean()),
                    "log_mse": float((logerror**2).mean()),
                    "log_error_acf1": float(logerror.corr(logerror.shift(1))),
                    "underprediction_mean_run": float(runs.mean()) if len(runs) else 0.0,
                    "underprediction_max_run": int(runs.max()) if len(runs) else 0,
                }
            )
    return pd.DataFrame(rows)


def decision_frames(daily: pd.DataFrame, forecasts: pd.DataFrame, config: dict) -> dict:
    frames = {}
    for symbol, group in daily.groupby("symbol"):
        market = group.sort_values("date").copy()
        market["prev_close"] = market.close.shift(1)
        market["valid_prev_close"] = market.valid_close.shift(1).eq(True)
        f = forecasts.loc[
            (forecasts.symbol == symbol) & (forecasts.model == config["primary_model"])
        ].copy()
        f = (
            f.merge(
                market[
                    [
                        "date",
                        "open",
                        "close",
                        "prev_close",
                        "valid_open",
                        "valid_close",
                        "valid_prev_close",
                    ]
                ],
                on="date",
                validate="one_to_one",
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        if not f[["valid_open", "valid_close", "valid_prev_close"]].all().all():
            raise ValueError(
                f"{symbol}: unobserved daily price boundary; execution rule needs explicit revision"
            )
        f["score"] = residual_score(
            f, config["correction_window_days"], config["observation_lag_days"]
        )
        frames[symbol] = f
    return frames


def run_study(
    data_dir: Path, output_dir: Path, config_path: Path, reuse_forecasts: bool = False
) -> dict:
    started = time.monotonic()
    source_dir = Path(__file__).parent
    source_hashes = {p.name: file_hash(p) for p in sorted(source_dir.glob("*.py"))}
    data_dir, output_dir, config_path = Path(data_dir), Path(output_dir), Path(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    # These are implemented study constants, not silent knobs with no effect.
    if config["observation_lag_days"] != 2 or config["seed"] != 42:
        raise ValueError("This version fixes observation lag=2 and model seed=42")
    daily_path = data_dir / "processed" / "daily.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"])
    if set(daily.symbol.unique()) != set(config["symbols"]):
        raise ValueError("Dataset symbols disagree with the frozen study config")
    snapshot = {
        "config_sha256": file_hash(config_path),
        "daily_sha256": file_hash(daily_path),
        "forecast_source_sha256": file_hash(Path(__file__).with_name("forecast.py")),
        "forecast_versions": {
            n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn"]
        },
    }
    predictions_path = output_dir / "forecasts.csv"
    prediction_meta = output_dir / "forecast_provenance.json"
    if reuse_forecasts:
        cached = json.loads(prediction_meta.read_text()) if prediction_meta.exists() else {}
        expected = (
            {**snapshot, "forecasts_sha256": file_hash(predictions_path)}
            if predictions_path.exists()
            else {}
        )
        if not expected or cached != expected:
            raise ValueError("Cannot reuse forecasts without matching data/config provenance")
        predictions = pd.read_csv(
            predictions_path, parse_dates=["date", "trained_through"], float_precision="round_trip"
        )
    else:
        print("Fitting monthly expanding forecasts (single CPU thread)...", flush=True)
        with threadpool_limits(limits=1):
            predictions = make_forecasts(daily, config["forecast_start"], config["forecast_end"])
        predictions.to_csv(predictions_path, index=False)
        prediction_meta.write_text(
            json.dumps({**snapshot, "forecasts_sha256": file_hash(predictions_path)}, indent=2)
            + "\n"
        )
    fm = forecast_metrics(predictions, config)
    fm.to_csv(output_dir / "forecast_metrics.csv", index=False)
    frames = decision_frames(daily, predictions, config)
    print("Selecting policy parameters using 2022–2023 only...", flush=True)
    chosen, trials = choose_policies(frames, config)
    trials.to_csv(output_dir / "validation_trials.csv", index=False)
    (output_dir / "selected_parameters.json").write_text(json.dumps(chosen, indent=2) + "\n")
    kwargs = {k: config[k] for k in ("target_vol", "annualization", "max_weight")}
    paths = []
    for symbol, frame in frames.items():
        for policy in ["raw", "smooth", "band", "buffer", "static"]:
            parameter = chosen["static_scales"][symbol] if policy == "static" else chosen[policy]
            for cost in config["cost_bps_grid"]:
                paths.append(policy_path(frame, policy, parameter, cost_bps=cost, **kwargs))
    paths = pd.concat(paths, ignore_index=True)
    paths.to_csv(output_dir / "decision_paths.csv", index=False)
    records = []
    for split, mask in [
        ("validation", paths.date <= pd.Timestamp(config["validation_end"])),
        ("test", paths.date >= pd.Timestamp(config["test_start"])),
    ]:
        for (symbol, policy, cost), g in paths.loc[mask].groupby(["symbol", "policy", "cost_bps"]):
            records.append(
                {
                    "split": split,
                    "symbol": symbol,
                    "policy": policy,
                    "cost_bps": cost,
                    **path_metrics(g, config["annualization"]),
                }
            )
    metrics = pd.DataFrame(records)
    metrics.to_csv(output_dir / "decision_metrics.csv", index=False)
    test = paths.loc[(paths.date >= pd.Timestamp(config["test_start"])) & paths.cost_bps.eq(0)]
    contrasts = []
    for comparator in ["static", "raw", "smooth", "band"]:
        paired = test.loc[test.policy.isin(["buffer", comparator])].pivot(
            index=["date", "symbol"], columns="policy", values="risk_loss"
        )
        difference = (paired.buffer - paired[comparator]).unstack("symbol")
        # Pooled daily mean only when BOTH assets observed. Preserve calendar.
        pooled = difference.mean(axis=1).where(difference.notna().all(axis=1))
        for symbol, vector in [(s, difference[s]) for s in difference] + [("pooled", pooled)]:
            for block in config["bootstrap_block_days"]:
                contrasts.append(
                    {
                        "comparison": f"buffer_minus_{comparator}",
                        "symbol": symbol,
                        "primary": comparator == "static" and symbol == "pooled",
                        **paired_block_interval(
                            vector, block, config["bootstrap_replicates"], config["seed"]
                        ),
                    }
                )
    intervals = pd.DataFrame(contrasts)
    intervals.to_csv(output_dir / "risk_comparisons.csv", index=False)
    sim = run_simulation(output_dir)
    for name in ["audit.json", "manifest.json"]:
        shutil.copy2(data_dir / name, output_dir / f"data_{name}")
    shutil.copy2(config_path, output_dir / "study_config.json")
    protocol_path = config_path.parent.parent / "docs" / "protocol.md"
    if protocol_path.exists():
        shutil.copy2(protocol_path, output_dir / "study_protocol.md")
    versions = {
        name: importlib.metadata.version(name)
        for name in ["volrisklab", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
    }
    if source_hashes != {p.name: file_hash(p) for p in sorted(source_dir.glob("*.py"))}:
        raise RuntimeError("Source changed during execution; rerun from a stable checkout")
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        **snapshot,
        "manifest_sha256": file_hash(data_dir / "manifest.json"),
        "protocol_sha256": file_hash(protocol_path) if protocol_path.exists() else None,
        "source_sha256": source_hashes,
        "elapsed_seconds": time.monotonic() - started,
        "method": "fixed historical walk-forward; no live trading",
    }
    (output_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    from .reporting import make_figures, write_report

    make_figures(daily, predictions, metrics, paths, output_dir, config)
    write_report(fm, metrics, intervals, chosen, output_dir, config)
    print(f"Finished. Reports: {output_dir.resolve()}", flush=True)
    return {"parameters": chosen, "provenance": provenance, "simulation": sim}
