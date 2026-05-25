#!/usr/bin/env python3
"""Create fixed paired validation crops matching the Kaggle 64 -> 256 task."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--meta-info", type=Path, required=True, help="Validation filename list from split_meta_info.py.")
    parser.add_argument("--output-root", type=Path, default=Path("data/val_crops"))
    parser.add_argument("--lr-size", type=int, default=64)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hr_dir = args.data_root / "train" / "hr"
    lr_dir = args.data_root / "train" / "lr"
    out_hr = args.output_root / "hr"
    out_lr = args.output_root / "lr"
    out_hr.mkdir(parents=True, exist_ok=True)
    out_lr.mkdir(parents=True, exist_ok=True)

    names = [line.strip() for line in args.meta_info.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(args.seed)
    hr_size = args.lr_size * args.scale

    written = 0
    for name in tqdm(names, desc="Writing validation crops"):
        lr = Image.open(lr_dir / name).convert("RGB")
        hr = Image.open(hr_dir / name).convert("RGB")
        if lr.width * args.scale != hr.width or lr.height * args.scale != hr.height:
            raise ValueError(f"Shape mismatch for {name}: LR={lr.size}, HR={hr.size}")
        if lr.width < args.lr_size or lr.height < args.lr_size:
            raise ValueError(f"LR image is smaller than requested crop size for {name}: {lr.size}")

        stem = Path(name).stem
        suffix = Path(name).suffix or ".png"
        for crop_idx in range(args.crops_per_image):
            x_lr = rng.randint(0, lr.width - args.lr_size)
            y_lr = rng.randint(0, lr.height - args.lr_size)
            x_hr = x_lr * args.scale
            y_hr = y_lr * args.scale

            lr_crop = lr.crop((x_lr, y_lr, x_lr + args.lr_size, y_lr + args.lr_size))
            hr_crop = hr.crop((x_hr, y_hr, x_hr + hr_size, y_hr + hr_size))
            out_name = f"{stem}_crop{crop_idx:02d}{suffix}"
            lr_crop.save(out_lr / out_name)
            hr_crop.save(out_hr / out_name)
            written += 1

    print(f"Wrote {written} paired crops to {args.output_root}")


if __name__ == "__main__":
    main()

