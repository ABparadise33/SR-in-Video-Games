#!/usr/bin/env python3
"""Split a clean meta_info file into train and validation filename lists."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-val", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise SystemExit(f"No filenames found in {args.input}")

    rng = random.Random(args.seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    val_count = max(1, round(len(shuffled) * args.val_ratio))
    if args.max_val is not None:
        val_count = min(val_count, args.max_val)

    val_names = sorted(shuffled[:val_count])
    train_names = sorted(shuffled[val_count:])

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text("".join(f"{name}\n" for name in train_names), encoding="utf-8")
    args.val_output.write_text("".join(f"{name}\n" for name in val_names), encoding="utf-8")

    print(f"train: {len(train_names)} -> {args.train_output}")
    print(f"val:   {len(val_names)} -> {args.val_output}")


if __name__ == "__main__":
    main()

