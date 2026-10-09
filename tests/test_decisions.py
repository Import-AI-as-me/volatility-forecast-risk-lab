import numpy as np
import pandas as pd
import pytest

from volrisklab.decisions import (
    paired_block_interval,
    policy_path,
    rebalance_cost,
    residual_score,
    run_lengths,
)


def frame(n=12):
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n),
            "forecast": 0.04 / 365,
            "rv": 0.04 / 365,
            "open": 100.0,
            "close": 100.0,
            "prev_close": 100.0,
            "score": 0.0,
        }
    )


def test_self_financing_cost_identity():
    for pre, target in [(0, 0.6), (0.8, 0.2), (0.5, 0.5), (1, 0)]:
        cost, notional = rebalance_cost(pre, target, 0.002)
        assert cost == pytest.approx(0.002 * abs(target * (1 - cost) - pre))
        assert notional == pytest.approx(abs(target * (1 - cost) - pre))


def test_drift_turnover_and_compounding():
    f = frame(2)
    f["close"] = [110, 110]
    f["open"] = [100, 110]
    f["prev_close"] = [100, 110]
    p = policy_path(f, "raw", 1, target_vol=0.1)
    assert p.weight.iloc[0] == pytest.approx(0.5)
    assert p.pretrade_weight.iloc[1] == pytest.approx(0.55 / 1.05)
    assert p.turnover.iloc[1] == pytest.approx(0.55 / 1.05 - 0.5)
    assert np.prod(1 + p.net_return) == pytest.approx(1.05)


def test_overnight_gap_accrues_to_previous_holdings():
    f = frame(2)
    f.loc[1, ["open", "close"]] = 120
    p = policy_path(f, "raw", 1, target_vol=0.1)
    assert p.net_return.iloc[1] == pytest.approx(0.1)
    assert p.pretrade_weight.iloc[1] == pytest.approx(0.6 / 1.1)


def test_causal_feedback_observation_cutoff():
    f = frame()
    f.loc[5, "rv"] *= 16
    s = residual_score(f)
    assert s.iloc[:7].eq(0).all()
    assert s.iloc[7] > 0
    changed = f.copy()
    changed.loc[8:, "rv"] *= 100
    np.testing.assert_allclose(s.iloc[:10], residual_score(changed).iloc[:10])


def test_missing_forecast_holds_shares_and_does_not_drop_day():
    f = frame(2)
    f.loc[0, "close"] = 110
    f.loc[1, ["open", "close", "prev_close"]] = 110
    f.loc[1, "forecast"] = np.nan
    p = policy_path(f, "raw", 1, target_vol=0.1)
    assert len(p) == 2
    assert p.turnover.iloc[1] == 0


def test_zero_correction_equals_raw_and_costs_reduce_wealth():
    f = frame()
    raw = policy_path(f, "raw", 1)
    buf = policy_path(f, "buffer", 0)
    np.testing.assert_allclose(raw.weight, buf.weight)
    charged = policy_path(f, "raw", 1, cost_bps=20)
    assert np.prod(1 + charged.net_return) < np.prod(1 + raw.net_return)


def test_run_lengths_and_degenerate_bootstrap():
    np.testing.assert_array_equal(run_lengths([1, 1, 0, 1, 0, 1, 1, 1]), [2, 1, 3])
    stats = paired_block_interval(np.zeros(30), block=7, repetitions=100)
    assert stats["mean_difference"] == stats["ci_low"] == stats["ci_high"] == 0
