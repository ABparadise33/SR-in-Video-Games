#!/usr/bin/env python3
"""Convert plain base64 rle strings into Kaggle's expected bytes repr strings."""

from __future__ import annotations

import argparse
import ast
import base64
import zlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_rle(value: str) -> str:
    text = str(value)
    if text.startswith("b'") or text.startswith('b"'):
        decoded = ast.literal_eval(text)
        if not isinstance(decoded, bytes):
            raise ValueError("rle bytes repr did not evaluate to bytes")
        zlib.decompress(base64.b64decode(decoded))
        return text

    decoded = text.encode("ascii")
    zlib.decompress(base64.b64decode(decoded))
    return str(decoded)


def main() -> None:
    args = parse_args()
    submission = pd.read_csv(args.input)
    required = {"id", "filename", "rle"}
    missing = required - set(submission.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")

    submission["rle"] = submission["rle"].map(normalize_rle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote fixed submission: {args.output}")


if __name__ == "__main__":
    main()

