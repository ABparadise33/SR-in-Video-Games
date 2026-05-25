#!/usr/bin/env python3
"""Plot loss and validation curves from train_hat_with_csv.py outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/training_curves.png"))
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

    if "l_pix" in train and train["l_pix"].notna().any():
        axes[0].plot(train["iter"], train["l_pix"], label="l_pix", color="tab:blue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if "psnr" in val and val["psnr"].notna().any():
        axes[1].plot(val["iter"], val["psnr"], marker="o", label="val PSNR", color="tab:green")
    axes[1].set_title("Validation PSNR")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("PSNR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if "ssim" in val and val["ssim"].notna().any():
        axes[2].plot(val["iter"], val["ssim"], marker="o", label="val SSIM", color="tab:orange")
    axes[2].set_title("Validation SSIM")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("SSIM")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

