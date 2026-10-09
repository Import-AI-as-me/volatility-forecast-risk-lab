"""Causal, calendar-aware daily realized-variance forecasts.

An exposure held on date ``t`` may use observations only through ``t - 2``.
This deliberately leaves a full calendar day between the final feature day and
the held day.  Log-HAR and its nonlinear counterpart share exactly the same
three features and are refitted monthly on expanding historical data.  No
feature, target, or missing observation is filled by interpolation.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

MODEL_NAMES = ("persistence", "EWMA10", "logHAR", "ML_HAR")
FEATURE_COLUMNS = ("log_rv_lag2", "log_rv_mean5_lag2", "log_rv_mean22_lag2")
FORECAST_FLOOR = 1e-10
OUTPUT_COLUMNS = ("date", "symbol", "model", "forecast", "rv", "return", "trained_through")


def _positive_rv(frame: pd.DataFrame) -> pd.Series:
    """Use positive, finite RV only when its source validation flag is true."""
    rv = pd.to_numeric(frame["rv"], errors="coerce").astype(float)
    valid = frame["valid_rv"].fillna(False).astype(bool)
    return rv.where(valid & np.isfinite(rv) & (rv > 0.0))


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return lag-two log-HAR features on a complete daily calendar.

    The input must have a sorted, unique, gapless daily DatetimeIndex. Missing
    measurements must be represented by rows with NaN or ``valid_rv=False``.
    Requiring all observations in each rolling window prevents gaps from being
    silently bridged. The weekly/monthly features are logs of mean variance,
    rather than means of log variance.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Feature input must have a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Feature dates must be unique and sorted")
    if len(frame) > 1 and not (frame.index.to_series().diff().iloc[1:] == timedelta(days=1)).all():
        raise ValueError("Feature input must include every calendar day")
    lagged = _positive_rv(frame).shift(2)
    return pd.DataFrame(
        {
            FEATURE_COLUMNS[0]: np.log(lagged),
            FEATURE_COLUMNS[1]: np.log(lagged.rolling(5, min_periods=5).mean()),
            FEATURE_COLUMNS[2]: np.log(lagged.rolling(22, min_periods=22).mean()),
        },
        index=frame.index,
    )


def _date(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def _calendar_frame(group: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Keep absent dates absent in value, rather than compressing the time axis."""
    group = group.sort_values("date").set_index("date")
    if not group.index.is_unique:
        raise ValueError("Each symbol must have at most one observation per date")
    calendar = pd.date_range(min(group.index.min(), start), max(group.index.max(), end), freq="D")
    frame = group.reindex(calendar)
    frame.index.name = "date"
    frame["valid_rv"] = frame["valid_rv"].eq(True)
    frame["rv"] = _positive_rv(frame)
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    return frame


def _fit_log_model(model, features: pd.DataFrame, target: pd.Series, train: pd.Series):
    """Fit log RV and a training-only Duan smearing correction."""
    x_train = features.loc[train]
    y_train = np.log(target.loc[train].to_numpy())
    model.fit(x_train, y_train)
    residual = y_train - model.predict(x_train)
    smearing = float(np.mean(np.exp(residual)))
    return model, smearing


def make_forecasts(
    daily: pd.DataFrame,
    forecast_start: str = "2022-01-01",
    forecast_end: str = "2025-12-31",
    min_train: int = 500,
) -> pd.DataFrame:
    """Build long-format variance forecasts without target or future leakage.

    Required columns are ``date, symbol, rv, valid_rv, return``. Dates are UTC
    calendar dates; each symbol is calendar-reindexed independently. Output
    includes every held date in the requested interval, for all four models,
    even if its target or forecast is missing. Invalid/nonpositive RV is NaN.

    HAR models are fit once per calendar month, at its first requested held
    date, using only valid targets dated at most two days before that date.
    Predictions use observed lagged features as they become available within
    the month; model parameters and smearing factors remain fixed until refit.
    Baselines do not fit parameters. Their ``trained_through`` is the date of
    the latest eligible observation (held date minus two); fitted models report
    the most recent actual target date in their monthly training set.

    EWMA10 uses alpha=2/11 (span=10), adjust=False, min_periods=10 and calendar
    gap weighting (ignore_na=False). It does not invent missing measurements;
    its output is masked whenever the latest eligible measurement is invalid.
    Log-HAR and ML_HAR both predict log variance and apply their own training-
    residual smearing factor to return to the variance scale. Predictions are
    bounded below by 1e-10. Hyperparameters are fixed; no search is performed.
    """
    required = {"date", "symbol", "rv", "valid_rv", "return"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if min_train < 1:
        raise ValueError("min_train must be positive")
    start, end = _date(forecast_start), _date(forecast_end)
    if start > end:
        raise ValueError("forecast_start must not exceed forecast_end")
    if daily.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = daily.copy()
    data["date"] = (
        pd.to_datetime(data["date"], utc=True, errors="raise").dt.tz_localize(None).dt.normalize()
    )
    if data["date"].isna().any() or data["symbol"].isna().any():
        raise ValueError("Dates and symbols cannot be missing")

    output = []
    held_dates = pd.date_range(start, end, freq="D", name="date")
    for symbol, group in data.groupby("symbol", sort=True):
        frame = _calendar_frame(group, start, end)
        target = frame["rv"]
        features = make_features(frame)
        complete_features = np.isfinite(features).all(axis=1)
        eligible_targets = target.notna() & complete_features

        forecasts = pd.DataFrame(index=held_dates, columns=MODEL_NAMES, dtype=float)
        trained = pd.DataFrame(index=held_dates, columns=MODEL_NAMES, dtype="datetime64[ns]")
        lagged = target.shift(2)
        persistence = lagged.reindex(held_dates)
        ewma = (
            lagged.ewm(span=10, adjust=False, min_periods=10, ignore_na=False)
            .mean()
            .where(lagged.notna())
        )
        forecasts["persistence"] = persistence.clip(lower=FORECAST_FLOOR)
        forecasts["EWMA10"] = ewma.reindex(held_dates).clip(lower=FORECAST_FLOOR)
        for name in ("persistence", "EWMA10"):
            trained[name] = pd.Series(held_dates - timedelta(days=2), index=held_dates).where(
                forecasts[name].notna()
            )

        for _, month_dates in pd.Series(held_dates, index=held_dates).groupby(
            held_dates.to_period("M")
        ):
            prediction_dates = pd.DatetimeIndex(month_dates.to_numpy())
            refit_date = prediction_dates[0]
            cutoff = refit_date - timedelta(days=2)
            train = eligible_targets & (frame.index <= cutoff)
            if int(train.sum()) < min_train:
                continue
            last_train_date = frame.index[train][-1]
            available_dates = prediction_dates[
                complete_features.reindex(prediction_dates).to_numpy()
            ]
            # Preserve the fitted model's provenance even on dates whose input
            # features are unavailable; their forecast remains NaN.
            trained.loc[prediction_dates, ["logHAR", "ML_HAR"]] = last_train_date
            if available_dates.empty:
                continue
            models = {
                "logHAR": LinearRegression(),
                "ML_HAR": HistGradientBoostingRegressor(
                    max_iter=100,
                    max_leaf_nodes=7,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    min_samples_leaf=30,
                    random_state=42,
                    early_stopping=False,
                ),
            }
            for name, model in models.items():
                fitted, smearing = _fit_log_model(model, features, target, train)
                prediction = np.exp(fitted.predict(features.loc[available_dates])) * smearing
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(
                        f"Nonfinite {name} forecast for {symbol} at {refit_date}"
                    )
                forecasts.loc[available_dates, name] = np.maximum(prediction, FORECAST_FLOOR)

        for name in MODEL_NAMES:
            output.append(
                pd.DataFrame(
                    {
                        "date": held_dates,
                        "symbol": symbol,
                        "model": name,
                        "forecast": forecasts[name].to_numpy(),
                        "rv": target.reindex(held_dates).to_numpy(),
                        "return": frame["return"].reindex(held_dates).to_numpy(),
                        "trained_through": trained[name].to_numpy(),
                    }
                )
            )

    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["date", "symbol", "model"])
        .reset_index(drop=True)
    )
