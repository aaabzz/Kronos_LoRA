"""Command-line entry point for the XtQuant downloader."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

from .config import XtQuantDownloadConfig
from .dataset_filter import filter_existing_dataset
from .downloader import XtQuantDownloader, normalize_universe
from .manifest import load_manifest
from .validator import validate_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one year of A-share 5-minute bars"
    )
    parser.add_argument(
        "action",
        choices=("plan", "download", "retry-failed", "validate", "filter-existing"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="full-market Parquet directory used by filter-existing",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--years", type=int, default=1)
    parser.add_argument("--download-batch-size", type=int, default=100)
    parser.add_argument("--read-batch-size", type=int, default=100)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--dividend-type", default="front")
    parser.add_argument("--exclude-beijing", action="store_true")
    parser.add_argument("--include-today", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--symbols-file",
        type=Path,
        action="append",
        dest="symbols_files",
        help=(
            "optional UTF-8 text file containing one symbol per line; repeat "
            "the option to use the cleaned union of multiple files"
        ),
    )
    parser.add_argument(
        "--symbol",
        help=(
            "single XtQuant A-share code, for example 600000.SH; cannot be "
            "combined with --symbols-file"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _load_symbols(paths: Path | Iterable[Path] | None) -> list[str] | None:
    """Read one or more files; downstream normalization forms the union."""

    if paths is None:
        return None
    requested_paths = [paths] if isinstance(paths, Path) else list(paths)
    symbols: list[str] = []
    for path in requested_paths:
        file_symbols = [
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not file_symbols:
            raise ValueError(f"symbols file is empty: {path}")
        symbols.extend(file_symbols)
    if not symbols:
        raise ValueError("no symbols files were provided")
    return symbols


def _symbol_source_paths(paths: Iterable[Path] | None) -> list[str]:
    return sorted({str(path.expanduser().resolve()) for path in (paths or [])})


def _requested_symbols(args: argparse.Namespace) -> tuple[list[str] | None, str]:
    """Resolve a direct symbol or symbol-file union through one normalizer."""
    if args.symbol is not None and args.symbols_files:
        raise ValueError("--symbol cannot be combined with --symbols-file")
    if args.symbol is not None:
        symbols = normalize_universe([args.symbol])
        if len(symbols) != 1:
            raise ValueError(
                "--symbol must be an A-share XtQuant code ending in .SH, .SZ or .BJ"
            )
        return symbols, "symbol"
    symbols = _load_symbols(args.symbols_files)
    return symbols, "symbols_file" if symbols is not None else "market"


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    options = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "lookback_years": args.years,
        "include_beijing": not args.exclude_beijing,
        "include_today": args.include_today,
        "download_batch_size": args.download_batch_size,
        "read_batch_size": args.read_batch_size,
        "retry_count": args.retry_count,
        "dividend_type": args.dividend_type,
        "resume": not args.no_resume,
    }
    if args.output_dir is not None:
        options["output_dir"] = args.output_dir
    config = XtQuantDownloadConfig(**options)

    if args.action == "validate":
        result = validate_output(config.output_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if result["status"] != "valid":
            raise SystemExit(1)
        return

    if args.action == "filter-existing":
        if (
            args.source_dir is None
            or args.output_dir is None
            or (not args.symbols_files and args.symbol is None)
        ):
            raise ValueError(
                "filter-existing requires --source-dir, --output-dir and either "
                "--symbol or --symbols-file"
            )
        symbols, universe_source = _requested_symbols(args)
        result = filter_existing_dataset(
            args.source_dir,
            args.output_dir,
            symbols or [],
            compression=config.compression,
            resume=config.resume,
            universe_sources=(
                [f"symbol:{symbols[0]}"]
                if universe_source == "symbol" and symbols
                else _symbol_source_paths(args.symbols_files)
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    if args.action == "retry-failed":
        manifest = load_manifest(config.output_dir / "download_manifest.json")
        if manifest is None:
            raise FileNotFoundError(
                f"download manifest does not exist under: {config.output_dir}"
            )
        saved_options = dict(manifest.get("config", {}))
        saved_options["output_dir"] = config.output_dir
        saved_options["resume"] = True
        if "fields" in saved_options:
            saved_options["fields"] = tuple(saved_options["fields"])
        config = XtQuantDownloadConfig(**saved_options)

    try:
        from xtquant import xtdata
    except ImportError as exc:
        raise SystemExit("xtquant is required: python -m pip install xtquant") from exc

    downloader = XtQuantDownloader(config, xtdata)
    symbols, universe_source = _requested_symbols(args)
    if args.action == "plan":
        result = downloader.plan(
            symbols=symbols,
            universe_source=universe_source,
        )
        if args.symbols_files:
            result["symbols_files"] = _symbol_source_paths(args.symbols_files)
        if universe_source == "symbol" and symbols:
            result["symbol"] = symbols[0]
    else:
        if args.action == "retry-failed" and symbols is not None:
            raise ValueError(
                "retry-failed uses the saved universe; omit --symbol/--symbols-file"
            )
        result = downloader.run(
            symbols=symbols,
            universe_source=universe_source,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
