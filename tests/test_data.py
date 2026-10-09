"""Synthetic checks for timestamp changes, missing intervals and file integrity."""

import hashlib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from volrisklab.data import _download_one, aggregate_klines, normalize_timestamps, verify_checksum


def make_bars(days=3):
    times = pd.date_range("2024-12-30", periods=288 * days, freq="5min", tz="UTC")
    close = 100 * np.exp(np.arange(len(times)) * 0.0001)
    return pd.DataFrame(
        {
            "open_time": times,
            "open": close / np.exp(0.0001),
            "high": close * 1.001,
            "low": close / 1.001,
            "close": close,
            "volume": np.ones(len(times)),
        }
    )


def test_timestamp_normalization_spans_vendor_unit_change():
    result = normalize_timestamps(pd.Series([1735689300000, 1735689600000000]))
    assert result.iloc[0] == pd.Timestamp("2024-12-31 23:55:00", tz="UTC")
    assert result.iloc[1] == pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    assert result.diff().iloc[1].total_seconds() == 300


def test_first_day_rv_missing_and_midnight_return_included():
    frame = make_bars()
    daily, audit = aggregate_klines(frame, "TEST")
    assert not daily.loc[0, "valid_rv"]
    assert np.isnan(daily.loc[0, "rv"])
    assert daily.loc[0, "rv_returns"] == 287
    assert daily.loc[1, "valid_rv"]
    assert daily.loc[1, "rv"] == pytest.approx(288 * 0.0001**2)
    assert daily.loc[1, "return"] == pytest.approx(np.exp(288 * 0.0001) - 1)
    assert audit["missing_bars"] == 0


def test_missing_bar_does_not_create_a_long_interval_return():
    frame = make_bars().drop(index=288 + 100)
    daily, audit = aggregate_klines(frame, "TEST")
    assert audit["missing_bars"] == 1
    assert audit["gap_intervals"] == 1
    assert daily.loc[1, "bars"] == 287
    assert daily.loc[1, "rv_returns"] == 286
    assert np.isnan(daily.loc[1, "rv"])
    assert np.isnan(daily.loc[1, "return"])
    assert daily.loc[2, "valid_rv"]


def test_missing_midnight_previous_close_invalidates_next_day_rv():
    frame = make_bars().drop(index=287)
    daily, _ = aggregate_klines(frame, "TEST")
    assert daily.loc[1, "bars"] == 288
    assert daily.loc[1, "rv_returns"] == 287
    assert not daily.loc[1, "valid_rv"]
    assert np.isfinite(daily.loc[1, "return"])


def test_calendar_includes_fully_missing_day():
    frame = make_bars()
    frame = frame.drop(index=range(288, 576))
    daily, audit = aggregate_klines(frame, "TEST")
    assert len(daily) == 3
    assert daily.loc[1, "bars"] == 0
    assert np.isnan(daily.loc[1, "rv"])
    assert not daily.loc[1, "valid_open"]
    assert not daily.loc[1, "valid_close"]
    assert not daily.loc[2, "valid_rv"]
    assert audit["missing_bars"] == 288


def test_boundary_flags_require_exact_utc_first_and_last_bar():
    frame = make_bars().drop(index=[288, 575])
    daily, audit = aggregate_klines(frame, "TEST")
    assert daily.loc[0, "valid_open"] and daily.loc[0, "valid_close"]
    assert not daily.loc[1, "valid_open"]
    assert not daily.loc[1, "valid_close"]
    assert daily.loc[2, "valid_open"] and daily.loc[2, "valid_close"]
    assert audit["invalid_open_days"] == 1
    assert audit["invalid_close_days"] == 1
    assert audit["invalid_day_examples"][1]["bars"] == 286


def test_duplicate_bars_are_rejected():
    frame = make_bars()
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_klines(pd.concat([frame, frame.iloc[[2]]]), "TEST")


def test_nonpositive_price_rejected_but_zero_volume_allowed():
    frame = make_bars()
    frame.loc[42, "volume"] = 0
    _, audit = aggregate_klines(frame, "TEST")
    assert audit["zero_volume_bars"] == 1
    frame.loc[42, "close"] = 0
    with pytest.raises(ValueError, match="nonpositive"):
        aggregate_klines(frame, "TEST")


def test_checksum_detects_corruption(tmp_path):
    path = tmp_path / "sample.zip"
    path.write_bytes(b"original data")
    digest = hashlib.sha256(b"original data").hexdigest()
    checksum = tmp_path / "sample.zip.CHECKSUM"
    checksum.write_text(f"{digest}  sample.zip\n")
    assert verify_checksum(path, checksum) == digest
    path.write_bytes(b"corrupted data")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_checksum(path, checksum)


def test_checksum_rejects_wrong_archive_name(tmp_path):
    path = tmp_path / "sample.zip"
    path.write_bytes(b"original data")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum = tmp_path / "sample.zip.CHECKSUM"
    checksum.write_text(f"{digest}  another.zip\n")
    with pytest.raises(ValueError, match="different archive"):
        verify_checksum(path, checksum)


def make_cached_archive(data_dir):
    archive = data_dir / "raw" / "TEST" / "TEST-5m-2024-01.zip"
    archive.parent.mkdir(parents=True)
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("TEST-5m-2024-01.csv", "synthetic fixture\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.CHECKSUM").write_text(f"{digest}  {archive.name}\n")
    return digest


def test_cached_vendor_revision_cannot_silently_change_pinned_dataset(tmp_path):
    make_cached_archive(tmp_path)
    with pytest.raises(ValueError, match="differs from existing manifest"):
        _download_one(tmp_path, "TEST", "2024-01", {"sha256": "0" * 64})


def test_inferred_cache_retrieval_provenance_survives_repeated_download(tmp_path):
    digest = make_cached_archive(tmp_path)
    first = _download_one(tmp_path, "TEST", "2024-01", {})
    second = _download_one(tmp_path, "TEST", "2024-01", first)
    assert first["sha256"] == digest
    assert first["retrieval_time_inferred_from_cache"]
    assert second["retrieval_time_inferred_from_cache"]
    assert first["retrieved_at_utc"] == second["retrieved_at_utc"]
