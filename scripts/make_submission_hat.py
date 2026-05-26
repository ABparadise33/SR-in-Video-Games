#!/usr/bin/env python3
"""Run HAT inference and write a Kaggle submission CSV."""

from __future__ import annotations

import argparse
import base64
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hat-root", type=Path, default=Path("external/HAT"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opt",
        type=Path,
        default=None,
        help="Optional HAT/BasicSR YAML. When set, build the network from network_g instead of hardcoded HAT-S.",
    )
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--test-lr", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submissions/hat_s_baseline.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", action="store_true", help="Use x8 flip/transpose test-time augmentation.")
    parser.add_argument("--rgb", action="store_true", help="Encode RGB instead of Kaggle's expected cv2/BGR order.")
    parser.add_argument("--bgr", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def encode(img: np.ndarray) -> str:
    img_to_encode = img.astype(np.uint8).flatten()
    img_to_encode = np.append(img_to_encode, -1)

    count = 1
    rle: list[int] = []
    for idx in range(1, img_to_encode.shape[0]):
        if img_to_encode[idx] == img_to_encode[idx - 1]:
            count += 1
            if count > 255:
                rle += [int(img_to_encode[idx - 1]), 255]
                count = 1
        else:
            rle += [int(img_to_encode[idx - 1]), count]
            count = 1

    compressed = zlib.compress(bytes(rle), zlib.Z_BEST_COMPRESSION)
    return str(base64.b64encode(compressed))


def load_network_options(opt: Path | None) -> dict:
    if opt is None:
        return {
            "upscale": 4,
            "in_chans": 3,
            "img_size": 64,
            "window_size": 16,
            "compress_ratio": 24,
            "squeeze_factor": 24,
            "conv_scale": 0.01,
            "overlap_ratio": 0.5,
            "img_range": 1.0,
            "depths": [6, 6, 6, 6, 6, 6],
            "embed_dim": 144,
            "num_heads": [6, 6, 6, 6, 6, 6],
            "mlp_ratio": 2,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv",
        }

    with opt.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    network_options = dict(config["network_g"])
    network_options.pop("type", None)
    return network_options


def build_hat_model(hat_root: Path, checkpoint: Path, device: str, opt: Path | None) -> torch.nn.Module:
    sys.path.insert(0, str(hat_root.resolve()))
    from hat.archs.hat_arch import HAT  # noqa: PLC0415

    model = HAT(**load_network_options(opt))
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_rgb(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor


def tta_forward(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.flip(x, dims=[-2, -1]),
        lambda x: x.transpose(-1, -2),
        lambda x: torch.flip(x.transpose(-1, -2), dims=[-1]),
        lambda x: torch.flip(x.transpose(-1, -2), dims=[-2]),
        lambda x: torch.flip(x.transpose(-1, -2), dims=[-2, -1]),
    ]
    inverses = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2]),
        lambda x: torch.flip(x, dims=[-2, -1]),
        lambda x: x.transpose(-1, -2),
        lambda x: torch.flip(x, dims=[-1]).transpose(-1, -2),
        lambda x: torch.flip(x, dims=[-2]).transpose(-1, -2),
        lambda x: torch.flip(x, dims=[-2, -1]).transpose(-1, -2),
    ]
    preds = [inv(model(aug(batch))) for aug, inv in zip(transforms, inverses, strict=True)]
    return torch.stack(preds, dim=0).mean(dim=0)


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    sample_submission = args.sample_submission or data_root / "sample_submission.csv"
    test_lr = args.test_lr or data_root / "test" / "lr"

    submission = pd.read_csv(sample_submission)
    filenames = submission["filename"].tolist()
    model = build_hat_model(args.hat_root, args.checkpoint, args.device, args.opt)

    encoded_by_name: dict[str, str] = {}
    with torch.no_grad():
        for names in tqdm(batched(filenames, args.batch_size), desc="Predicting"):
            batch = torch.stack([load_rgb(test_lr / name) for name in names]).to(args.device)
            pred = tta_forward(model, batch) if args.tta else model(batch)
            pred = pred.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for name, image in zip(names, pred, strict=True):
                if not args.rgb:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                encoded_by_name[name] = encode(image)

    submission["rle"] = [encoded_by_name[name] for name in filenames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote submission: {args.output}")


if __name__ == "__main__":
    main()
