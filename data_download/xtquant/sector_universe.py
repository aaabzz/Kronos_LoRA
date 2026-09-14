"""Export a reproducible XtQuant sector constituent snapshot."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .downloader import normalize_universe, universe_digest
from .manifest import save_json


def refresh_sector_data(xtdata_module: Any) -> None:
    updater = getattr(xtdata_module, "download_sector_data", None)
    if callable(updater):
        updater()


def list_sector_names(xtdata_module: Any) -> list[str]:
    """Return the exact sector names advertised by the local XtQuant client."""

    refresh_sector_data(xtdata_module)
    getter = getattr(xtdata_module, "get_sector_list", None)
    if not callable(getter):
        raise RuntimeError("this XtQuant version does not provide get_sector_list()")
    return sorted(
        {
            value.strip()
            for value in (getter() or [])
            if isinstance(value, str) and value.strip()
        }
    )


def export_sector_universe(
    xtdata_module: Any,
    sector_names: str | Iterable[str],
    output_dir: Path,
    *,
    include_beijing: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save the union of one or more sectors as an immutable snapshot."""

    refresh_sector_data(xtdata_module)
    requested_names = [sector_names] if isinstance(sector_names, str) else list(sector_names)
    names = list(
        dict.fromkeys(
            name.strip()
            for name in requested_names
            if isinstance(name, str) and name.strip()
        )
    )
    if not names:
        raise ValueError("at least one non-empty sector name is required")

    membership: dict[str, list[str]] = {}
    sector_symbol_counts: dict[str, int] = {}
    for sector_name in names:
        sector_symbols = normalize_universe(
            xtdata_module.get_stock_list_in_sector(sector_name) or []
        )
        if not include_beijing:
            sector_symbols = [
                symbol for symbol in sector_symbols if not symbol.endswith(".BJ")
            ]
        if not sector_symbols:
            raise RuntimeError(
                f"sector does not exist or contains no A shares: {sector_name}"
            )
        sector_symbol_counts[sector_name] = len(sector_symbols)
        for symbol in sector_symbols:
            membership.setdefault(symbol, []).append(sector_name)
    symbols = sorted(membership)

    output_dir = Path(output_dir).expanduser().resolve()
    symbols_path = output_dir / "symbols.txt"
    metadata_path = output_dir / "sector_metadata.json"
    membership_path = output_dir / "sector_membership.json"
    if not overwrite and any(
        path.exists() for path in (symbols_path, metadata_path, membership_path)
    ):
        raise FileExistsError(
            f"sector snapshot already exists under {output_dir}; use --overwrite "
            "or select a new dated directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = "\n".join(symbols) + "\n"
    temporary = symbols_path.with_suffix(".txt.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, symbols_path)

    result: dict[str, Any] = {
        "version": 2,
        "kind": "xtquant_sector_snapshot",
        # Preserve the original scalar field for single-sector consumers.
        "sector_name": names[0] if len(names) == 1 else None,
        "sector_names": names,
        "sector_count": len(names),
        "sector_symbol_counts": sector_symbol_counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "membership_semantics": "current_constituents_at_generation_time",
        "include_beijing": include_beijing,
        "symbol_count": len(symbols),
        "source_symbol_assignments": sum(sector_symbol_counts.values()),
        "overlap_assignments": sum(sector_symbol_counts.values()) - len(symbols),
        "universe_sha256": universe_digest(symbols),
        "symbols_file": str(symbols_path),
        "membership_file": str(membership_path),
    }
    save_json(membership_path, membership)
    save_json(metadata_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List XtQuant sectors or export a fixed constituent snapshot."
    )
    parser.add_argument("action", choices=("list", "export"))
    parser.add_argument(
        "--sector-name",
        action="append",
        dest="sector_names",
        help="exact XtQuant sector name; repeat to export the cleaned union",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contains", help="optional substring filter for list")
    parser.add_argument("--exclude-beijing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from xtquant import xtdata
    except ImportError as exc:
        raise SystemExit("xtquant is required: python -m pip install xtquant") from exc

    if args.action == "list":
        names = list_sector_names(xtdata)
        if args.contains:
            names = [name for name in names if args.contains in name]
        print(json.dumps({"count": len(names), "sectors": names}, ensure_ascii=False, indent=2))
        return

    if not args.sector_names or args.output_dir is None:
        raise ValueError("export requires --sector-name and --output-dir")
    result = export_sector_universe(
        xtdata,
        args.sector_names,
        args.output_dir,
        include_beijing=not args.exclude_beijing,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
