#!/usr/bin/env python3
"""Run HAT training and save parsed training/validation metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


TRAIN_FIELDS = [
    "epoch",
    "iter",
    "lrs",
    "time",
    "data_time",
    "l_pix",
    "l_g_pix",
    "l_g_percep",
    "l_g_style",
    "l_g_gan",
    "l_d_real",
    "l_d_fake",
    "out_d_real",
    "out_d_fake",
]
VAL_FIELDS = [
    "iter",
    "dataset",
    "psnr",
    "best_psnr",
    "best_psnr_iter",
    "ssim",
    "best_ssim",
    "best_ssim_iter",
]

TRAIN_ITER_RE = re.compile(
    r"epoch:\s*([0-9]+),\s*iter:\s*([0-9,\s]+?),\s*lr",
    re.IGNORECASE,
)
LR_RE = re.compile(r"(?:lrs:\s*\[?[\(\[]?|lr:\()([0-9.eE+-]+)")
TIME_RE = re.compile(r"time \(data\):\s*([0-9.]+)\s*\(([0-9.]+)\)")
SCALAR_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*):\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
)
VAL_START_RE = re.compile(r"Validation\s+(.+)$")
VAL_METRIC_RE = re.compile(
    r"#\s*(psnr|ssim):\s*([0-9.]+)\s+Best:\s*([0-9.]+)\s*@\s*([0-9,]+)\s*iter",
    re.IGNORECASE,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hat-root", type=Path, default=Path("external/HAT"))
    parser.add_argument("--opt", required=True, help="Path to HAT option YAML, relative to hat-root or absolute.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--log-name", default=None)
    return parser.parse_known_args()


def open_writer(path: Path, fields: list[str]) -> tuple[csv.DictWriter, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(file_obj, fieldnames=fields)
    if path.stat().st_size == 0:
        writer.writeheader()
        file_obj.flush()
    return writer, file_obj


def parse_train_line(line: str) -> dict[str, str] | None:
    match = TRAIN_ITER_RE.search(line)
    if match is None:
        return None

    row: dict[str, str] = {
        "epoch": match.group(1),
        "iter": match.group(2).replace(",", "").replace(" ", ""),
    }
    lr_match = LR_RE.search(line)
    if lr_match is not None:
        row["lrs"] = lr_match.group(1)

    time_match = TIME_RE.search(line)
    if time_match is not None:
        row["time"] = time_match.group(1)
        row["data_time"] = time_match.group(2)

    for key, value in SCALAR_RE.findall(line):
        if key in TRAIN_FIELDS and key not in row:
            row[key] = value
    return row


def flush_val(writer: csv.DictWriter, file_obj: object, pending: dict[str, str] | None) -> None:
    if pending and ("psnr" in pending or "ssim" in pending):
        writer.writerow({field: pending.get(field, "") for field in VAL_FIELDS})
        file_obj.flush()


def main() -> int:
    args, extra_args = parse_args()
    hat_root = args.hat_root.resolve()
    opt = Path(args.opt)
    opt_for_hat = str(opt if opt.is_absolute() else opt)
    log_name = args.log_name or Path(args.opt).stem
    log_dir = args.log_dir.resolve()

    terminal_log = log_dir / f"{log_name}.log"
    train_csv = log_dir / f"{log_name}_train_metrics.csv"
    val_csv = log_dir / f"{log_name}_val_metrics.csv"

    train_writer, train_file = open_writer(train_csv, TRAIN_FIELDS)
    val_writer, val_file = open_writer(val_csv, VAL_FIELDS)
    terminal_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "hat/train.py", "-opt", opt_for_hat, *extra_args]
    print("+", " ".join(cmd), flush=True)
    print(f"terminal log: {terminal_log}", flush=True)
    print(f"train csv:    {train_csv}", flush=True)
    print(f"val csv:      {val_csv}", flush=True)

    pending_val: dict[str, str] | None = None
    last_train_iter = ""
    with terminal_log.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=hat_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

            train_row = parse_train_line(line)
            if train_row is not None:
                flush_val(val_writer, val_file, pending_val)
                pending_val = None
                last_train_iter = train_row.get("iter", last_train_iter)
                train_writer.writerow({field: train_row.get(field, "") for field in TRAIN_FIELDS})
                train_file.flush()
                continue

            val_start = VAL_START_RE.search(line)
            if val_start is not None:
                flush_val(val_writer, val_file, pending_val)
                pending_val = {"iter": last_train_iter, "dataset": val_start.group(1).strip()}
                continue

            val_metric = VAL_METRIC_RE.search(line)
            if val_metric is not None:
                metric = val_metric.group(1).lower()
                pending_val = pending_val or {"dataset": ""}
                pending_val[metric] = val_metric.group(2)
                pending_val[f"best_{metric}"] = val_metric.group(3)
                pending_val[f"best_{metric}_iter"] = val_metric.group(4).replace(",", "")

        return_code = process.wait()

    flush_val(val_writer, val_file, pending_val)
    train_file.close()
    val_file.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
