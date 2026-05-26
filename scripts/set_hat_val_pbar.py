#!/usr/bin/env python3
"""Toggle validation progress bars in a HAT/BasicSR YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--enable", action="store_true", help="Enable validation pbar. Default disables it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data.setdefault("val", {})["pbar"] = bool(args.enable)
    args.config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"Set val.pbar={bool(args.enable)} in {args.config}")


if __name__ == "__main__":
    main()
