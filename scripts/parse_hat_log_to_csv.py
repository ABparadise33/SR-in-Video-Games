#!/usr/bin/env python3
"""Parse an existing HAT terminal log into train/validation metric CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from train_hat_with_csv import (
    TRAIN_FIELDS,
    VAL_FIELDS,
    VAL_METRIC_RE,
    VAL_START_RE,
    flush_val,
    open_writer,
    parse_train_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_writer, train_file = open_writer(args.train_csv, TRAIN_FIELDS)
    val_writer, val_file = open_writer(args.val_csv, VAL_FIELDS)

    pending_val: dict[str, str] | None = None
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        train_row = parse_train_line(line)
        if train_row is not None:
            flush_val(val_writer, val_file, pending_val)
            pending_val = None
            train_writer.writerow({field: train_row.get(field, "") for field in TRAIN_FIELDS})
            continue

        val_start = VAL_START_RE.search(line)
        if val_start is not None:
            flush_val(val_writer, val_file, pending_val)
            pending_val = {"dataset": val_start.group(1).strip()}
            continue

        val_metric = VAL_METRIC_RE.search(line)
        if val_metric is not None:
            metric = val_metric.group(1).lower()
            pending_val = pending_val or {"dataset": ""}
            pending_val[metric] = val_metric.group(2)
            pending_val[f"best_{metric}"] = val_metric.group(3)
            pending_val[f"best_{metric}_iter"] = val_metric.group(4).replace(",", "")
            pending_val["iter"] = val_metric.group(4).replace(",", "")

    flush_val(val_writer, val_file, pending_val)
    train_file.close()
    val_file.close()
    print(f"Wrote {args.train_csv}")
    print(f"Wrote {args.val_csv}")


if __name__ == "__main__":
    main()

