"""Timing and missing-data checks for the forecasting layer."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from volrisklab.forecast import FEATURE_COLUMNS, MODEL_NAMES, make_features, make_forecasts


def sample_daily(periods=820):
    rng = np.random.default_rng(17)
    dates = pd.date_range("2019-01-01", periods=periods, freq="D")
    log_rv = -8.0 + 0.3 * np.sin(np.arange(periods) / 30.0) + rng.normal(0, 0.1, periods)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "TEST",
            "rv": np.exp(log_rv),
            "valid_rv": True,
            "return": rng.normal(0, 0.01, periods),
        }
    )


def test_features_use_two_calendar_day_lag_and_arithmetic_mean():
    dates = pd.date_range("2020-01-01", periods=30)
    frame = pd.DataFrame({"rv": np.arange(1, 31, dtype=float), "valid_rv": True}, index=dates)
    features = make_features(frame)
    assert features.loc[dates[25], FEATURE_COLUMNS[0]] == pytest.approx(np.log(24.0))
    assert features.loc[dates[25], FEATURE_COLUMNS[1]] == pytest.approx(
        np.log(np.mean(np.arange(20, 25)))
    )
    assert features.loc[dates[25], FEATURE_COLUMNS[2]] == pytest.approx(
        np.log(np.mean(np.arange(3, 25)))
    )
    assert features.loc[dates[:23], FEATURE_COLUMNS[2]].isna().all()


@pytest.mark.parametrize("bad_value", [np.nan, 0.0, -1.0, np.inf])
def test_invalid_rv_is_not_interpolated_or_used_in_rolling_features(bad_value):
    frame = sample_daily(50).set_index("date")
    bad_date = frame.index[25]
    frame.loc[bad_date, "rv"] = bad_value
    features = make_features(frame)
    assert pd.isna(features.loc[bad_date + timedelta(days=2), FEATURE_COLUMNS[0]])
    assert (
        features.loc[
            bad_date + timedelta(days=2) : bad_date + timedelta(days=23), FEATURE_COLUMNS[2]
        ]
        .isna()
        .all()
    )


def test_invalid_source_flag_is_honored():
    frame = sample_daily(50).set_index("date")
    frame.loc[frame.index[25], "valid_rv"] = False
    features = make_features(frame)
    assert features.loc[frame.index[27]].isna().all()


def test_features_reject_compressed_calendar():
    frame = sample_daily(40).set_index("date").drop(pd.Timestamp("2019-01-12"))
    with pytest.raises(ValueError, match="every calendar day"):
        make_features(frame)


def test_target_and_future_changes_cannot_affect_forecasts():
    daily = sample_daily()
    held = pd.Timestamp("2021-02-10")
    original = make_forecasts(daily, "2021-02-01", "2021-02-12")
    altered = daily.copy()
    # Even the immediately preceding day is not yet eligible for held-day
    # features. Mutating it also catches an accidental lag-one implementation.
    future = altered["date"] >= held - timedelta(days=1)
    altered.loc[future, "rv"] *= 50.0
    altered.loc[future, "return"] = -0.9
    changed = make_forecasts(altered, "2021-02-01", "2021-02-12")
    cols = ["date", "symbol", "model", "forecast", "trained_through"]
    pd.testing.assert_frame_equal(
        original.loc[original["date"] <= held, cols].reset_index(drop=True),
        changed.loc[changed["date"] <= held, cols].reset_index(drop=True),
    )


def test_monthly_training_cutoff_and_fixed_within_month():
    result = make_forecasts(sample_daily(), "2021-01-01", "2021-02-15")
    learned = result[result["model"].isin(["logHAR", "ML_HAR"])]
    jan = learned[learned["date"].dt.month == 1]
    feb = learned[learned["date"].dt.month == 2]
    assert jan["trained_through"].eq(pd.Timestamp("2020-12-30")).all()
    assert feb["trained_through"].eq(pd.Timestamp("2021-01-30")).all()
    assert (result["trained_through"] <= result["date"] - timedelta(days=2)).all()
    assert np.isfinite(result["forecast"]).all()
    assert (result["forecast"] > 0).all()


def test_missing_target_and_absent_date_remain_in_output():
    daily = sample_daily()
    missing_date = pd.Timestamp("2021-02-03")
    daily = daily.loc[daily["date"] != missing_date]
    result = make_forecasts(daily, "2021-02-01", "2021-02-05")
    assert len(result) == 5 * len(MODEL_NAMES)
    absent = result[result["date"] == missing_date]
    assert absent["rv"].isna().all()
    assert absent["return"].isna().all()
    assert absent["forecast"].notna().all()
    affected = result[result["date"] == missing_date + timedelta(days=2)]
    assert affected["forecast"].isna().all()


def test_insufficient_training_preserves_dates_without_fitting():
    result = make_forecasts(sample_daily(100), "2019-03-01", "2019-03-05")
    learned = result[result["model"].isin(["logHAR", "ML_HAR"])]
    assert len(result) == 20
    assert learned["forecast"].isna().all()
    assert learned["trained_through"].isna().all()
