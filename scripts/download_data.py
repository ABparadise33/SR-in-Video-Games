#!/usr/bin/env python3
"""Download the Kaggle competition data with KaggleHub."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition",
        default="super-resolution-in-video-games",
        help="Kaggle competition slug.",
    )
    parser.add_argument(
        "--link",
        type=Path,
        default=None,
        help="Optional symlink path to create inside this repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import kagglehub

    path = Path(kagglehub.competition_download(args.competition)).resolve()
    print(f"Path to competition files: {path}")

    if args.link is not None:
        link = args.link
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.exists() or link.is_symlink():
            print(f"Link already exists: {link.resolve()}")
        else:
            link.symlink_to(path, target_is_directory=True)
            print(f"Created symlink: {link} -> {path}")


if __name__ == "__main__":
    main()

