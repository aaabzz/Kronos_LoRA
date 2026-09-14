"""Download the pinned Kronos model bundle from Hugging Face.

The revisions below are the same commits used by the repository regression
tests. Files are downloaded into sibling directories under ``pretrained`` so
later runs can load them without contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


PRETRAINED_DIR = Path(__file__).resolve().parent

MODEL_SPECS = {
    "small": {
        "repo_id": "NeoQuasar/Kronos-small",
        "revision": "901c26c1332695a2a8f243eb2f37243a37bea320",
        "local_dir": PRETRAINED_DIR / "Kronos-small",
    },
    "base": {
        "repo_id": "NeoQuasar/Kronos-base",
        "revision": "2b554741eca47781b64468546e77fef3e85130e6",
        "local_dir": PRETRAINED_DIR / "Kronos-base",
    },
}

TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
TOKENIZER_DIR = PRETRAINED_DIR / "Kronos-Tokenizer-base"

MANIFEST_PATH = PRETRAINED_DIR / "download_manifest.json"


def download_repository(repo_id: str, revision: str, destination: Path) -> str:
    """Download one pinned Hugging Face repository into destination."""
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}@{revision} -> {destination}")
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=destination,
    )


def write_manifest(downloaded_paths: dict[str, str]) -> None:
    manifest = {
        "models": {
            name: {
                "repo_id": spec["repo_id"],
                "revision": spec["revision"],
                "local_path": str(spec["local_dir"]),
                "snapshot_path": downloaded_paths[name],
            }
            for name, spec in MODEL_SPECS.items()
            if name in downloaded_paths
        },
        "tokenizer": {
            "repo_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "local_path": str(TOKENIZER_DIR),
            "snapshot_path": downloaded_paths["tokenizer"],
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pinned Kronos model and tokenizer snapshots."
    )
    parser.add_argument(
        "--only",
        choices=("all", "models", "model", "small", "base", "tokenizer"),
        default="all",
        help=(
            "Choose repositories to download. 'model' is retained as an alias "
            "for 'base' (default: all)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded_paths: dict[str, str] = {}

    selected_models: list[str] = []
    if args.only in ("all", "models"):
        selected_models = list(MODEL_SPECS)
    elif args.only in "small":
        selected_models = ["small"]
    elif args.only == ("model", "base"):
        selected_models = ["base"]

    for name in selected_models:
        spec = MODEL_SPECS[name]
        downloaded_paths[name] = download_repository(
            str(spec["repo_id"]),
            str(spec["revision"]),
            Path(spec["local_dir"]),
        )

    if args.only in ("all", "tokenizer"):
        downloaded_paths["tokenizer"] = download_repository(
            TOKENIZER_ID, TOKENIZER_REVISION, TOKENIZER_DIR
        )

    # Write a complete manifest only when all repositories were requested.
    if args.only == "all":
        write_manifest(downloaded_paths)

    print("Download completed.")


if __name__ == "__main__":
    main()
