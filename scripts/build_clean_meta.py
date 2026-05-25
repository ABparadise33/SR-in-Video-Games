#!/usr/bin/env python3
"""Build a BasicSR/HAT meta_info file after PSNR-based pair cleaning."""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hr-dir", type=Path, default=None)
    parser.add_argument("--lr-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=18.0)
    parser.add_argument("--check-size", type=int, nargs=2, default=(256, 256), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--output", type=Path, default=Path("meta_info/train_clean_psnr18.txt"))
    parser.add_argument("--rejects", type=Path, default=Path("meta_info/rejected_psnr18.csv"))
    parser.add_argument("--summary", type=Path, default=Path("meta_info/clean_psnr18_summary.json"))
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: process the first N pairs.")
    return parser.parse_args()


def list_images(folder: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def psnr_rgb(hr_path: Path, lr_path: Path, check_size: tuple[int, int]) -> float:
    hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
    lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
    if hr is None:
        raise ValueError(f"Could not read HR image: {hr_path}")
    if lr is None:
        raise ValueError(f"Could not read LR image: {lr_path}")

    width, height = check_size
    hr = cv2.resize(hr, (width, height), interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(lr, (width, height), interpolation=cv2.INTER_CUBIC)
    mse = np.mean((hr.astype(np.float32) - lr.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(float(mse)))


def score_one(item: tuple[str, str, str, tuple[int, int]]) -> tuple[str, float, str | None]:
    filename, hr_path, lr_path, check_size = item
    try:
        return filename, psnr_rgb(Path(hr_path), Path(lr_path), check_size), None
    except Exception as exc:  # noqa: BLE001
        return filename, float("-inf"), str(exc)


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    hr_dir = args.hr_dir or data_root / "train" / "hr"
    lr_dir = args.lr_dir or data_root / "train" / "lr"

    hr_images = list_images(hr_dir)
    lr_images = list_images(lr_dir)
    filenames = sorted(set(hr_images) & set(lr_images))
    if args.limit is not None:
        filenames = filenames[: args.limit]
    if not filenames:
        raise SystemExit(f"No paired images found in {hr_dir} and {lr_dir}")

    tasks = [(name, str(hr_images[name]), str(lr_images[name]), tuple(args.check_size)) for name in filenames]
    rows: list[tuple[str, float, str | None]] = []

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for row in tqdm(executor.map(score_one, tasks), total=len(tasks), desc="Scoring pairs"):
                rows.append(row)
    else:
        for task in tqdm(tasks, desc="Scoring pairs"):
            rows.append(score_one(task))

    clean = [(name, value) for name, value, err in rows if err is None and value >= args.threshold]
    rejected = [(name, value, err) for name, value, err in rows if err is not None or value < args.threshold]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{name}\n" for name, _ in clean), encoding="utf-8")

    args.rejects.parent.mkdir(parents=True, exist_ok=True)
    with args.rejects.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "psnr", "error"])
        writer.writerows(rejected)

    finite_scores = [value for _, value, err in rows if err is None and math.isfinite(value)]
    summary = {
        "data_root": str(data_root),
        "hr_dir": str(hr_dir),
        "lr_dir": str(lr_dir),
        "threshold": args.threshold,
        "check_size": list(args.check_size),
        "total_pairs": len(rows),
        "kept": len(clean),
        "rejected": len(rejected),
        "min_psnr": min(finite_scores) if finite_scores else None,
        "mean_psnr": float(np.mean(finite_scores)) if finite_scores else None,
        "max_psnr": max(finite_scores) if finite_scores else None,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

