"""CLI for downloading, reproducing, and inspecting the experiment."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="volrisklab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="Download and validate the frozen public dataset")
    data.add_argument("--data-dir", type=Path, default=Path("data"))
    data.add_argument("--config", type=Path, default=Path("configs/study.json"))
    data.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path("reports/generated/data_manifest.json"),
        help="Seed a fresh data directory with the published archive hashes",
    )
    run = sub.add_parser("run", help="Run fixed-protocol real-data research and simulation")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    run.add_argument("--config", type=Path, default=Path("configs/study.json"))
    run.add_argument(
        "--reuse-forecasts", action="store_true", help="Require matching data/config hashes"
    )
    sim = sub.add_parser("simulate", help="Offline controlled experiment; no market-data download")
    sim.add_argument("--output-dir", type=Path, default=Path("reports/simulation"))
    args = parser.parse_args()
    if args.command == "data":
        import json
        import shutil

        from .data import build_daily, download_dataset

        config = json.loads(args.config.read_text())
        if not (args.data_dir / "manifest.json").exists() and args.reference_manifest.exists():
            reference = json.loads(args.reference_manifest.read_text())
            if (
                reference["start_month"] != config["data_start"]
                or reference["end_month"] != config["data_end"]
                or set(reference["symbols"]) != set(config["symbols"])
            ):
                raise ValueError(
                    "Reference manifest does not match study config; choose the correct snapshot"
                )
            args.data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.reference_manifest, args.data_dir / "manifest.json")
        download_dataset(
            args.data_dir, config["data_start"], config["data_end"], tuple(config["symbols"])
        )
        build_daily(args.data_dir)
    elif args.command == "run":
        from .analysis import run_study

        run_study(args.data_dir, args.output_dir, args.config, args.reuse_forecasts)
    else:
        from .simulation import run_simulation

        result = run_simulation(args.output_dir)
        print(f"Simulation written to {args.output_dir.resolve()}")
        print(result["summary"])


if __name__ == "__main__":
    main()
