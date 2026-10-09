"""Check mathematical identities rather than a model's fitted performance."""

import csv
import math

import numpy as np
import pytest

from volrisklab.simulation import (
    DEFAULT_ERROR_AMPLITUDE,
    DEFAULT_STEPS,
    TARGET_ANNUAL_VOL,
    _run_lengths,
    compute_scenarios,
    run_simulation,
)


def test_same_error_multiset_and_invariant_pointwise_losses():
    paths, summaries = compute_scenarios()
    expected_errors = [-DEFAULT_ERROR_AMPLITUDE] * (DEFAULT_STEPS // 2) + [
        DEFAULT_ERROR_AMPLITUDE
    ] * (DEFAULT_STEPS // 2)
    reference = summaries[0]
    for summary in summaries:
        rows = [row for row in paths if row["scenario"] == summary["scenario"]]
        assert sorted(row["log_variance_error"] for row in rows) == expected_errors
        assert summary["mean_qlike"] == pytest.approx(
            math.cosh(DEFAULT_ERROR_AMPLITUDE) - 1, abs=1e-14
        )
        assert summary["mean_squared_log_variance_error"] == pytest.approx(
            DEFAULT_ERROR_AMPLITUDE**2, abs=1e-14
        )
        assert summary["risk_tracking_mse"] == pytest.approx(
            reference["risk_tracking_mse"], abs=1e-14
        )
        assert summary["overshoot_fraction"] == 0.5
        # The invariant is pointwise and does not depend on a particular order.
        risks = np.array([row["annualized_risk_proxy"] for row in rows])
        assert np.mean((risks - TARGET_ANNUAL_VOL) ** 2) == pytest.approx(
            summary["risk_tracking_mse"]
        )


def test_block_run_lengths_and_exact_turnover_formula():
    _, summaries = compute_scenarios()
    a = DEFAULT_ERROR_AMPLITUDE
    high_weight = 0.75 * math.exp(a / 2)
    low_weight = 0.75 * math.exp(-a / 2)
    turnover_values = []
    for summary, block_length in zip(summaries, (1, 5, 20)):
        switches = DEFAULT_STEPS // block_length - 1
        expected_turnover = high_weight + switches * (high_weight - low_weight)
        assert summary["longest_overshoot_run"] == block_length
        assert summary["mean_overshoot_run"] == block_length
        assert summary["overshoot_run_count"] == DEFAULT_STEPS // (2 * block_length)
        assert summary["number_of_weight_switches"] == switches
        assert summary["initial_turnover"] == pytest.approx(high_weight)
        assert summary["total_turnover"] == pytest.approx(expected_turnover)
        assert summary["turnover_from_switch_formula"] == pytest.approx(expected_turnover)
        turnover_values.append(summary["total_turnover"])
    assert turnover_values[0] > turnover_values[1] > turnover_values[2]


def test_run_lengths_count_endpoints_and_empty_cases():
    np.testing.assert_array_equal(_run_lengths(np.array([1, 1, 0, 1])), [2, 1])
    np.testing.assert_array_equal(_run_lengths(np.array([0, 0])), [])
    np.testing.assert_array_equal(_run_lengths(np.array([1, 1])), [2])


def test_capping_does_not_break_permutation_invariance():
    _, summaries = compute_scenarios(error_amplitude=1.0)
    reference = summaries[0]
    for summary in summaries:
        assert summary["weight_at_negative_error"] == 1.0
        assert summary["mean_qlike"] == pytest.approx(reference["mean_qlike"])
        assert summary["risk_tracking_mse"] == pytest.approx(reference["risk_tracking_mse"])
        assert summary["total_turnover"] == pytest.approx(summary["turnover_from_switch_formula"])


@pytest.mark.parametrize("n_steps", [0, 39, 42, 40.5])
def test_reject_incomplete_or_invalid_cycles(n_steps):
    with pytest.raises(ValueError):
        compute_scenarios(n_steps=n_steps)


def test_run_simulation_writes_self_contained_outputs(tmp_path):
    result = run_simulation(tmp_path)
    assert result["parameters"]["initial_turnover_included"] is True
    assert result["parameters"]["terminal_liquidation_included"] is False
    assert (tmp_path / "simulation.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with (tmp_path / "simulation_paths.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * DEFAULT_STEPS
    with (tmp_path / "simulation_summary.csv").open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 3
    assert [int(row["longest_overshoot_run"]) for row in summaries] == [1, 5, 20]
