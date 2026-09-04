from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Kaggle username slug.")
    parser.add_argument("--slug", default="r1-grpo-kaggle", help="Kaggle kernel slug.")
    parser.add_argument("--title", default="R1 GRPO Kaggle", help="Kaggle kernel title.")
    parser.add_argument("--private", action="store_true", help="Create a private Kaggle kernel.")
    parser.add_argument(
        "--internet",
        action="store_true",
        help="Enable internet in Kaggle. Required to download models/datasets unless mirrored as Kaggle inputs.",
    )
    args = parser.parse_args()

    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": args.title,
        "code_file": "kaggle_entry.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if args.private else "false",
        "enable_gpu": "true",
        "enable_internet": "true" if args.internet else "false",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    output_path = Path("kernel-metadata.json")
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Kernel id: {metadata['id']}")


if __name__ == "__main__":
    main()

