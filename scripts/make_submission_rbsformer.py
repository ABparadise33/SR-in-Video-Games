#!/usr/bin/env python3
"""Run RGB-adapted RBSFormer inference and write a Kaggle submission."""

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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srvg.models import RBSFormerRGB


def encode(img: np.ndarray) -> str:
    img_to_encode = img.astype(np.uint8).flatten()
    img_to_encode = np.append(img_to_encode, -1)
    cnt, rle = 1, []
    for i in range(1, img_to_encode.shape[0]):
        if img_to_encode[i] == img_to_encode[i - 1]:
            cnt += 1
            if cnt > 255:
                rle += [int(img_to_encode[i - 1]), 255]
                cnt = 1
        else:
            rle += [int(img_to_encode[i - 1]), cnt]
            cnt = 1
    return str(base64.b64encode(zlib.compress(bytes(rle), zlib.Z_BEST_COMPRESSION)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/rbsformer_rgb.csv"))
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--channels", type=int, default=60)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--no-residual-upsample", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", action="store_true", help="Use x8 flip/transpose test-time augmentation.")
    return parser.parse_args()


def load_rgb(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


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


def main() -> None:
    args = parse_args()
    model = RBSFormerRGB(
        channels=args.channels,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        residual_upsample=not args.no_residual_upsample,
    ).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    sample = args.sample_submission or args.data_root / "sample_submission.csv"
    submission = pd.read_csv(sample)
    filenames = submission["filename"].tolist()
    test_lr = args.data_root / "test" / "lr"

    encoded: dict[str, str] = {}
    with torch.no_grad():
        for names in tqdm(batched(filenames, args.batch_size), desc="Predicting"):
            batch = torch.stack([load_rgb(test_lr / name) for name in names]).to(args.device)
            pred = tta_forward(model, batch) if args.tta else model(batch)
            pred = pred.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            for name, image_rgb in zip(names, pred, strict=True):
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                encoded[name] = encode(image_bgr)

    submission["rle"] = [encoded[name] for name in filenames]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
