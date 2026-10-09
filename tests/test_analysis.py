"""Cross-model sample fairness and validation/test separation checks."""

import numpy as np
import pandas as pd
import pytest

from volrisklab.analysis import forecast_metrics
from volrisklab.decisions import choose_policies, policy_path, residual_score


def test_forecast_metrics_use_common_dates_and_keep_missing_day_as_run_break():
    dates = pd.date_range("2024-01-01", periods=5)
    predictions = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "TEST",
                "model": model,
                "forecast": 1.0 if model == "a" else 2.0,
                "rv": rv,
            }
            for date, rv in zip(dates, [4.0, 6.0, 4.0, 8.0, 10.0], strict=True)
            for model in ("a", "b")
        ]
    )
    predictions.loc[(predictions.date == dates[2]) & (predictions.model == "b"), "forecast"] = (
        np.nan
    )
    metrics = forecast_metrics(
        predictions,
        {
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
        },
    ).set_index("model")
    assert metrics["valid_days"].eq(4).all()
    assert metrics["calendar_days"].eq(5).all()
    # Excluding a date for one model excludes it for both. It must break runs,
    # not concatenate two separate two-day episodes into a four-day episode.
    assert metrics["underprediction_max_run"].eq(2).all()
    assert metrics["underprediction_mean_run"].eq(2).all()
    common_rv = np.array([4.0, 6.0, 8.0, 10.0])
    assert metrics.loc["a", "qlike"] == pytest.approx(np.mean(common_rv - np.log(common_rv) - 1))
    assert metrics.loc["b", "qlike"] == pytest.approx(
        np.mean(common_rv / 2 - np.log(common_rv / 2) - 1)
    )


def _selection_fixture():
    dates = pd.date_range("2022-01-01", periods=45)
    frames = {}
    for symbol, level in (("A", 0.0004), ("B", 0.0008)):
        frame = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "forecast": level * (1 + 0.3 * np.cos(np.arange(45) / 3)),
                "rv": level * (1 + 0.5 * np.sin(np.arange(45) / 4)) ** 2,
                "open": 100.0,
                "close": 100.0,
                "prev_close": 100.0,
            }
        )
        frame["score"] = residual_score(frame)
        frames[symbol] = frame
    config = {
        "validation_end": "2022-01-25",
        "target_vol": 0.15,
        "annualization": 365,
        "max_weight": 1.0,
        "smoothing_alpha_grid": [0.25, 1.0],
        "no_trade_band_grid": [0.0, 0.03],
        "correction_gamma_grid": [0.0, 0.5, 1.0],
        "selection_turnover_penalty": 0.1,
    }
    return frames, config


def test_policy_selection_and_static_scale_are_invariant_to_future_data():
    frames, config = _selection_fixture()
    chosen, trials = choose_policies(frames, config)
    changed = {}
    for symbol, frame in frames.items():
        altered = frame.copy()
        future = altered.date > pd.Timestamp(config["validation_end"])
        altered.loc[future, "rv"] *= 100
        altered.loc[future, "forecast"] *= 0.01
        altered.loc[future, ["open", "close", "prev_close"]] *= 20
        altered["score"] = residual_score(altered)
        changed[symbol] = altered
    changed_chosen, changed_trials = choose_policies(changed, config)
    assert changed_chosen == chosen
    pd.testing.assert_frame_equal(changed_trials, trials)


def test_static_scale_matches_validation_exposure_only():
    frames, config = _selection_fixture()
    chosen, _ = choose_policies(frames, config)
    kwargs = {key: config[key] for key in ("target_vol", "annualization", "max_weight")}
    for symbol, frame in frames.items():
        validation = frame.loc[frame.date <= pd.Timestamp(config["validation_end"])]
        buffer = policy_path(validation, "buffer", chosen["buffer"], **kwargs)
        static = policy_path(validation, "static", chosen["static_scales"][symbol], **kwargs)
        assert static.weight.mean() == pytest.approx(buffer.weight.mean(), abs=1e-12)
