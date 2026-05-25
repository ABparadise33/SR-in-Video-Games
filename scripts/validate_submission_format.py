#!/usr/bin/env python3
"""Validate Kaggle submission columns and rle byte-string formatting."""

from __future__ import annotations

import argparse
import ast
import base64
import zlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=10)
    return parser.parse_args()


def check_rle(value: str) -> int:
    decoded = ast.literal_eval(str(value))
    if not isinstance(decoded, bytes):
        raise ValueError("rle did not evaluate to bytes")
    raw = zlib.decompress(base64.b64decode(decoded))
    return len(raw)


def main() -> None:
    args = parse_args()
    submission = pd.read_csv(args.submission)
    required = ["id", "filename", "rle"]
    if list(submission.columns) != required:
        raise SystemExit(f"Expected columns {required}, got {list(submission.columns)}")

    if args.sample_submission is not None:
        sample = pd.read_csv(args.sample_submission)
        if len(submission) != len(sample):
            raise SystemExit(f"Row count mismatch: {len(submission)} != {len(sample)}")
        if submission["filename"].tolist() != sample["filename"].tolist():
            raise SystemExit("Filename order differs from sample submission")

    for idx, value in enumerate(submission["rle"].head(args.max_rows)):
        check_rle(value)
        print(f"row {idx}: ok")
    print(f"Validated {min(args.max_rows, len(submission))} rows from {args.submission}")


if __name__ == "__main__":
    main()

