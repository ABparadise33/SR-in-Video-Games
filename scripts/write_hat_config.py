#!/usr/bin/env python3
"""Generate the HAT-S baseline training config for this competition."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hat-root", type=Path, default=Path("external/HAT"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrain", type=Path, required=True)
    parser.add_argument("--meta-info", type=Path, default=None)
    parser.add_argument("--val-root", type=Path, default=None, help="Optional validation crop root with hr/ and lr/.")
    parser.add_argument("--val-meta-info", type=Path, default=None)
    parser.add_argument("--val-freq", type=float, default=2000)
    parser.add_argument("--val-crop-border", type=int, default=0)
    parser.add_argument("--val-y-channel", action="store_true")
    parser.add_argument("--save-val-img", action="store_true")
    parser.add_argument("--name", default="train_HAT-S_gamesr_baseline")
    parser.add_argument("--total-iter", type=int, default=24000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--save-freq", type=float, default=2000)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--prefetch-mode", default=None, choices=["cuda", "cpu"])
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional source HAT YAML. Defaults to HAT-S SRx4 fine-tune config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path. Defaults to HAT options/train/<name>.yml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hat_root = args.hat_root
    template = args.template or hat_root / "options" / "train" / "train_HAT-S_SRx4_finetune_from_SRx2.yml"
    output = args.output or hat_root / "options" / "train" / f"{args.name}.yml"

    with template.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = args.data_root.resolve()
    config["name"] = args.name
    config["num_gpu"] = 1
    config["pbar"] = True
    datasets = {
        "train": {
            "name": "SR in Video Games",
            "type": "PairedImageDataset",
            "dataroot_gt": str(data_root / "train" / "hr"),
            "dataroot_lq": str(data_root / "train" / "lr"),
            "io_backend": {"type": "disk"},
            "gt_size": 256,
            "use_hflip": True,
            "use_rot": True,
            "use_shuffle": True,
            "num_worker_per_gpu": args.workers,
            "batch_size_per_gpu": args.batch_size,
            "dataset_enlarge_ratio": 1,
            "pin_memory": True,
        }
    }
    config["datasets"] = datasets
    if args.meta_info is not None:
        config["datasets"]["train"]["meta_info_file"] = str(args.meta_info.resolve())
    if args.prefetch_mode is not None:
        config["datasets"]["train"]["prefetch_mode"] = args.prefetch_mode

    if args.val_root is not None:
        val_root = args.val_root.resolve()
        datasets["val"] = {
            "name": "SR in Video Games Val Crops",
            "type": "PairedImageDataset",
            "dataroot_gt": str(val_root / "hr"),
            "dataroot_lq": str(val_root / "lr"),
            "io_backend": {"type": "disk"},
        }
        if args.val_meta_info is not None:
            datasets["val"]["meta_info_file"] = str(args.val_meta_info.resolve())

    config["path"]["pretrain_network_g"] = str(args.pretrain.resolve())
    config["path"]["strict_load_g"] = False
    config["train"]["total_iter"] = args.total_iter
    config["logger"]["print_freq"] = args.print_freq
    config["logger"]["save_checkpoint_freq"] = args.save_freq
    config["val"] = {
        "val_freq": args.val_freq,
        "save_img": args.save_val_img,
        "pbar": True,
        "metrics": {
            "psnr": {
                "type": "calculate_psnr",
                "crop_border": args.val_crop_border,
                "test_y_channel": args.val_y_channel,
                "better": "higher",
            },
            "ssim": {
                "type": "calculate_ssim",
                "crop_border": args.val_crop_border,
                "test_y_channel": args.val_y_channel,
                "better": "higher",
            },
        },
    }
    if args.val_root is None:
        config.pop("val", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print(f"Wrote HAT config: {output}")


if __name__ == "__main__":
    main()
