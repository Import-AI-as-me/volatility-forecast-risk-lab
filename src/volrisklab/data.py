"""Verified public Binance data acquisition and conservative UTC aggregation.

Raw market data is downloaded locally, not redistributed with this project.
Archive provenance and hashes are recorded in ``data/manifest.json``. Missing
five-minute bars are never filled and returns never bridge a missing interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
INTERVAL = "5m"
BAR_SECONDS = 300
BARS_PER_DAY = 288
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum(path: Path, checksum_path: Path) -> str:
    """Verify the vendor SHA256 sidecar, rejecting malformed or mismatched data."""
    fields = checksum_path.read_text(encoding="utf-8").strip().split()
    if not fields or not re.fullmatch(r"[0-9a-fA-F]{64}", fields[0]):
        raise ValueError(f"Malformed checksum: {checksum_path}")
    expected = fields[0].lower()
    if len(fields) > 1 and Path(fields[1].lstrip("*")).name != path.name:
        raise ValueError(f"Checksum names a different archive: {checksum_path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path.name}: {actual} != {expected}")
    return actual


def _fetch(url: str, destination: Path) -> None:
    """Fetch with verified TLS; system curl is a certificate-store fallback."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "VolRiskLab/0.1 research"})
            with (
                urlopen(request, context=context, timeout=60) as response,
                temporary.open("wb") as output,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            temporary.replace(destination)
            return
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(attempt + 1)
    # macOS system curl can use the system trust store. Never use --insecure.
    try:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "2",
                "--connect-timeout",
                "20",
                "--max-time",
                "120",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        temporary.replace(destination)
    except (OSError, subprocess.CalledProcessError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {last_error}; {error}") from error


def _months(start: str, end: str) -> list[str]:
    for value in (start, end):
        if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
            raise ValueError("Month arguments must have YYYY-MM format")
    if start > end:
        raise ValueError("start must not follow end")
    return [str(month) for month in pd.period_range(start, end, freq="M")]


def _download_one(data_dir: Path, symbol: str, month: str, previous: dict) -> dict:
    name = f"{symbol}-{INTERVAL}-{month}.zip"
    archive = data_dir / "raw" / symbol / name
    checksum = archive.with_suffix(".zip.CHECKSUM")
    url = f"{BASE_URL}/{symbol}/{INTERVAL}/{name}"
    cached = archive.exists() and checksum.exists()
    if not cached:
        _fetch(url + ".CHECKSUM", checksum)
        _fetch(url, archive)
    # Do not silently replace a corrupt cache: fail with the exact offending file.
    digest = verify_checksum(archive, checksum)
    if previous.get("sha256") and previous["sha256"] != digest:
        raise ValueError(
            f"Archive differs from existing manifest: {archive.name}. "
            "Preserve the existing manifest and use a new data directory for a revised dataset."
        )
    with ZipFile(archive) as zipped:
        corrupt_member = zipped.testzip()
        if corrupt_member:
            raise ValueError(f"Corrupt archive member {corrupt_member} in {archive}")
    prior_time = (
        previous.get("retrieved_at_utc") if cached and previous.get("sha256") == digest else None
    )
    return {
        "symbol": symbol,
        "month": month,
        "interval": INTERVAL,
        "path": archive.relative_to(data_dir).as_posix(),
        "url": url,
        "checksum_url": url + ".CHECKSUM",
        "sha256": digest,
        "bytes": archive.stat().st_size,
        "retrieved_at_utc": prior_time or _utc_now(),
        "retrieval_time_inferred_from_cache": (
            bool(previous.get("retrieval_time_inferred_from_cache", False))
            if prior_time
            else bool(cached)
        ),
    }


def download_dataset(
    data_dir: Path,
    start: str = "2019-01",
    end: str = "2025-12",
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT"),
    workers: int = 4,
) -> Path:
    """Download monthly archives and write an ordered, verified manifest.

    Existing cache files are checked against their saved vendor checksums.
    An existing manifest pins archive hashes. Use a new data directory to obtain
    later vendor revisions without changing an existing experiment's dataset.
    """
    data_dir = Path(data_dir)
    if not symbols or any(not re.fullmatch(r"[A-Z0-9]+", symbol) for symbol in symbols):
        raise ValueError("symbols must be uppercase alphanumeric market symbols")
    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must not contain duplicates")
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    months = _months(start, end)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = {row["path"]: row for row in json.loads(manifest_path.read_text())["archives"]}
    archives = []
    jobs = [(symbol, month) for symbol in symbols for month in months]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one,
                data_dir,
                symbol,
                month,
                previous.get(f"raw/{symbol}/{symbol}-{INTERVAL}-{month}.zip", {}),
            ): (symbol, month)
            for symbol, month in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            archives.append(record)
            if index % 12 == 0 or index == len(jobs):
                print(f"Verified {index}/{len(jobs)} monthly archives", flush=True)
    manifest = {
        "schema_version": 1,
        "source": "Binance Public Data (spot klines)",
        "source_documentation": "https://github.com/binance/binance-public-data",
        "dataset_terms": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
        "created_at_utc": _utc_now(),
        "start_month": start,
        "end_month": end,
        "symbols": list(symbols),
        "interval": INTERVAL,
        "archives": sorted(archives, key=lambda row: (row["symbol"], row["month"])),
    }
    temporary = manifest_path.with_suffix(".json.part")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def normalize_timestamps(values: pd.Series) -> pd.Series:
    """Normalize mixed millisecond/microsecond Unix timestamps by magnitude."""
    numeric = pd.to_numeric(values, errors="raise")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError("Missing or nonfinite timestamps")
    if ((numeric % 1) != 0).any() or (numeric < 100_000_000_000).any():
        raise ValueError("Expected integer Unix millisecond/microsecond timestamps")
    microseconds = numeric.where(numeric >= 100_000_000_000_000, numeric * 1000)
    return pd.to_datetime(microseconds.astype("int64"), unit="us", utc=True)


def read_klines(archive: Path) -> pd.DataFrame:
    with ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist() if name.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Expected one CSV in {archive}")
        with zipped.open(names[0]) as stream:
            frame = pd.read_csv(stream, header=None)
    if len(frame.columns) != len(COLUMNS):
        raise ValueError(f"Expected 12 kline columns in {archive}, found {len(frame.columns)}")
    frame.columns = COLUMNS
    frame["open_time"] = normalize_timestamps(frame["open_time"])
    return frame[["open_time", "open", "high", "low", "close", "volume"]]


def aggregate_klines(
    frame: pd.DataFrame,
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate a single symbol; return daily observations and a coverage audit.

    A valid RV day requires 288 bars AND 288 finite adjacent-close returns,
    including the preceding UTC day's final close. The first sample day has no
    previous close and therefore has missing RV. Daily simple return uses the
    first open and last close and is missing on incomplete days.
    """
    if frame.empty:
        raise ValueError(f"No observations for {symbol}")
    frame = frame.copy()
    if not pd.api.types.is_datetime64_any_dtype(frame["open_time"]):
        frame["open_time"] = normalize_timestamps(frame["open_time"])
    else:
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame = frame.sort_values("open_time").reset_index(drop=True)
    duplicates = int(frame["open_time"].duplicated().sum())
    if duplicates:
        raise ValueError(f"{symbol}: {duplicates} duplicate bar timestamps")
    if (frame["open_time"].dt.floor("5min") != frame["open_time"]).any():
        raise ValueError(f"{symbol}: bar timestamps are not aligned to five-minute UTC grid")
    price_columns = ["open", "high", "low", "close"]
    for column in price_columns + ["volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{symbol}: nonfinite {column}")
    if (frame[price_columns] <= 0).any().any():
        raise ValueError(f"{symbol}: nonpositive OHLC price")
    if (frame["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume")
    if (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError(f"{symbol}: inconsistent OHLC range")

    differences = frame["open_time"].diff().dt.total_seconds()
    adjacent = differences.eq(BAR_SECONDS)
    log_returns = np.log(frame["close"]).diff().where(adjacent)
    frame["squared_return"] = log_returns.pow(2)
    frame["date"] = frame["open_time"].dt.floor("D")
    daily = frame.groupby("date", sort=True).agg(
        bars=("open_time", "size"),
        rv_returns=("squared_return", "count"),
        rv=("squared_return", "sum"),
        open=("open", "first"),
        close=("close", "last"),
        volume=("volume", "sum"),
        first_bar_time=("open_time", "first"),
        last_bar_time=("open_time", "last"),
    )
    first_day = pd.Timestamp(start, tz="UTC") if start else frame["date"].min()
    last_day = pd.Timestamp(end, tz="UTC") if end else frame["date"].max()
    if frame["date"].min() < first_day or frame["date"].max() > last_day:
        raise ValueError(f"{symbol}: observations outside requested daily calendar")
    daily = daily.reindex(pd.date_range(first_day, last_day, freq="D", name="date"))
    daily[["bars", "rv_returns"]] = daily[["bars", "rv_returns"]].fillna(0).astype(int)
    complete = daily["bars"].eq(BARS_PER_DAY)
    daily["valid_open"] = daily["first_bar_time"].eq(daily.index)
    # All timestamps already passed the exact five-minute-grid check above.
    daily["valid_close"] = daily["last_bar_time"].dt.hour.eq(23) & daily[
        "last_bar_time"
    ].dt.minute.eq(55)
    daily["valid_rv"] = complete & daily["rv_returns"].eq(BARS_PER_DAY)
    daily["rv"] = daily["rv"].where(daily["valid_rv"])
    daily["return"] = (daily["close"] / daily["open"] - 1).where(complete)
    daily["volume"] = daily["volume"].where(complete)
    daily["symbol"] = symbol
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.tz_localize(None)
    gaps = frame.loc[differences.gt(BAR_SECONDS), "open_time"]
    audit = {
        "symbol": symbol,
        "raw_rows": len(frame),
        "days": len(daily),
        "expected_bars": int(len(daily) * BARS_PER_DAY),
        "start_date": first_day.date().isoformat(),
        "end_date": last_day.date().isoformat(),
        "duplicate_timestamps": duplicates,
        "zero_volume_bars": int(frame["volume"].eq(0).sum()),
        "missing_bars": int(len(daily) * BARS_PER_DAY - len(frame)),
        "gap_intervals": len(gaps),
        "max_gap_seconds": int(differences.max()) if len(frame) > 1 else None,
        "first_gap_timestamps_utc": [stamp.isoformat() for stamp in gaps.iloc[:20]],
        "incomplete_days": int((~complete).sum()),
        "invalid_rv_days": int((~daily["valid_rv"]).sum()),
        "valid_rv_days": int(daily["valid_rv"].sum()),
        "invalid_open_days": int((~daily["valid_open"]).sum()),
        "invalid_close_days": int((~daily["valid_close"]).sum()),
        "invalid_day_examples": [
            {
                "date": row.date.date().isoformat(),
                "bars": int(row.bars),
                "rv_returns": int(row.rv_returns),
                "valid_open": bool(row.valid_open),
                "valid_close": bool(row.valid_close),
                "valid_rv": bool(row.valid_rv),
            }
            for row in daily.loc[~daily["valid_rv"]].head(30).itertuples()
        ],
        "first_day_has_previous_close": bool(adjacent.iloc[0]),
    }
    return daily[
        [
            "date",
            "symbol",
            "rv",
            "return",
            "valid_rv",
            "bars",
            "rv_returns",
            "volume",
            "open",
            "close",
            "valid_open",
            "valid_close",
        ]
    ], audit


def build_daily(data_dir: Path) -> pd.DataFrame:
    """Build only the files named by the manifest, rechecking their hashes."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    first_day = pd.Period(manifest["start_month"], freq="M").start_time.date().isoformat()
    last_day = pd.Period(manifest["end_month"], freq="M").end_time.date().isoformat()
    daily_frames, audits = [], []
    for symbol in manifest["symbols"]:
        frames = []
        records = [record for record in manifest["archives"] if record["symbol"] == symbol]
        for record in sorted(records, key=lambda item: item["month"]):
            archive = data_dir / record["path"]
            digest = verify_checksum(archive, archive.with_suffix(".zip.CHECKSUM"))
            if digest != record["sha256"]:
                raise ValueError(f"Archive differs from manifest: {archive}")
            frames.append(read_klines(archive))
        daily, audit = aggregate_klines(
            pd.concat(frames, ignore_index=True), symbol, first_day, last_day
        )
        daily_frames.append(daily)
        audits.append(audit)
        print(
            f"{symbol}: {audit['raw_rows']:,} bars; {audit['valid_rv_days']:,} valid RV days; "
            f"{audit['missing_bars']:,} missing bars",
            flush=True,
        )
    result = (
        pd.concat(daily_frames, ignore_index=True)
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    output = data_dir / "processed" / "daily.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, date_format="%Y-%m-%d")
    audit_payload = {
        "created_at_utc": _utc_now(),
        "manifest_sha256": sha256_file(data_dir / "manifest.json"),
        "daily_csv_sha256": sha256_file(output),
        "rv_definition": "Sum of 288 adjacent five-minute squared close-to-close log returns, grouped by UTC bar-open date; no gap bridging.",
        "return_definition": "Last close / first open - 1 within a complete UTC day.",
        "missing_policy": "RV missing unless all 288 bars and adjacent returns are present; daily return and volume missing on incomplete days; no imputation.",
        "symbols": audits,
    }
    (data_dir / "audit.json").write_text(
        json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=["download", "build", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--start", default="2019-01")
    parser.add_argument("--end", default="2025-12")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action in ("download", "all"):
        download_dataset(args.data_dir, args.start, args.end, tuple(args.symbols), args.workers)
    if args.action in ("build", "all"):
        build_daily(args.data_dir)


if __name__ == "__main__":
    main()
