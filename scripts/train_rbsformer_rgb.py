#!/usr/bin/env python3
"""Train an RGB-adapted RBSFormer baseline for SR in Video Games."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srvg.models import RBSFormerRGB


class PairedGameDataset(Dataset):
    def __init__(
        self,
        gt_dir: Path,
        lq_dir: Path,
        meta_info: Path,
        lr_size: int = 64,
        scale: int = 4,
        augment: bool = True,
    ) -> None:
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.names = [line.strip() for line in meta_info.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.lr_size = lr_size
        self.hr_size = lr_size * scale
        self.scale = scale
        self.augment = augment
        if not self.names:
            raise ValueError(f"No filenames found in {meta_info}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        lq = cv2.imread(str(self.lq_dir / name), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(self.gt_dir / name), cv2.IMREAD_COLOR)
        if lq is None or gt is None:
            raise FileNotFoundError(name)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        h, w = lq.shape[:2]
        x = random.randint(0, w - self.lr_size)
        y = random.randint(0, h - self.lr_size)
        xh, yh = x * self.scale, y * self.scale
        lq = lq[y : y + self.lr_size, x : x + self.lr_size]
        gt = gt[yh : yh + self.hr_size, xh : xh + self.hr_size]

        if self.augment:
            if random.random() < 0.5:
                lq = np.flip(lq, axis=1)
                gt = np.flip(gt, axis=1)
            if random.random() < 0.5:
                lq = np.flip(lq, axis=0)
                gt = np.flip(gt, axis=0)
            k = random.randint(0, 3)
            if k:
                lq = np.rot90(lq, k)
                gt = np.rot90(gt, k)

        lq = torch.from_numpy(np.ascontiguousarray(lq)).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float() / 255.0
        return lq, gt


class ValCropDataset(Dataset):
    def __init__(self, root: Path) -> None:
        self.gt_dir = root / "hr"
        self.lq_dir = root / "lr"
        self.names = sorted(path.name for path in self.lq_dir.iterdir() if path.is_file())
        if not self.names:
            raise ValueError(f"No validation crops found in {self.lq_dir}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        lq = cv2.cvtColor(cv2.imread(str(self.lq_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(str(self.gt_dir / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lq = torch.from_numpy(lq).permute(2, 0, 1).float() / 255.0
        gt = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
        return lq, gt


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps**2).mean()


def frequency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    return (pred_fft - target_fft).abs().mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred.clamp(0, 1), target, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def ssim_simple(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1, c2 = 0.01**2, 0.03**2
    pred = pred.clamp(0, 1)
    mu_x = pred.mean(dim=(-1, -2))
    mu_y = target.mean(dim=(-1, -2))
    sigma_x = ((pred - mu_x[..., None, None]) ** 2).mean(dim=(-1, -2))
    sigma_y = ((target - mu_y[..., None, None]) ** 2).mean(dim=(-1, -2))
    sigma_xy = ((pred - mu_x[..., None, None]) * (target - mu_y[..., None, None])).mean(dim=(-1, -2))
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--meta-info", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, default=Path("experiments/rbsformer_rgb"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--total-iter", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--lr-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=60)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--freq-loss-weight", type=float, default=0.0)
    parser.add_argument("--no-residual-upsample", action="store_true")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--val-freq", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for lq, gt in loader:
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            psnrs.append(psnr(pred, gt))
            ssims.append(ssim_simple(pred, gt))
    model.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.exp_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_set = PairedGameDataset(
        args.data_root / "train" / "hr",
        args.data_root / "train" / "lr",
        args.meta_info,
        lr_size=args.lr_size,
    )
    val_set = ValCropDataset(args.val_root)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    model = RBSFormerRGB(
        channels=args.channels,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        residual_upsample=not args.no_residual_upsample,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=args.lr * 0.01)
    charb = CharbonnierLoss()

    start_iter, best_psnr = 0, -1.0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_iter = state["iter"]
        best_psnr = state.get("best_psnr", best_psnr)

    train_csv = args.exp_dir / "train_metrics.csv"
    val_csv = args.exp_dir / "val_metrics.csv"
    with train_csv.open("a", newline="", encoding="utf-8") as train_f, val_csv.open("a", newline="", encoding="utf-8") as val_f:
        train_writer = csv.DictWriter(train_f, fieldnames=["iter", "lr", "l_charb", "l_freq", "loss"])
        val_writer = csv.DictWriter(val_f, fieldnames=["iter", "psnr", "ssim", "best_psnr"])
        if train_csv.stat().st_size == 0:
            train_writer.writeheader()
        if val_csv.stat().st_size == 0:
            val_writer.writeheader()

        iteration = start_iter
        progress = tqdm(total=args.total_iter, initial=start_iter, desc="Training RBSFormerRGB")
        while iteration < args.total_iter:
            for lq, gt in train_loader:
                iteration += 1
                lq, gt = lq.to(device), gt.to(device)
                pred = model(lq)
                l_charb = charb(pred, gt)
                l_freq = frequency_loss(pred, gt)
                loss = l_charb + args.freq_loss_weight * l_freq

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if iteration % args.print_freq == 0:
                    row = {
                        "iter": iteration,
                        "lr": optimizer.param_groups[0]["lr"],
                        "l_charb": l_charb.item(),
                        "l_freq": l_freq.item(),
                        "loss": loss.item(),
                    }
                    train_writer.writerow(row)
                    train_f.flush()
                    tqdm.write(
                        f"iter={iteration} lr={row['lr']:.3e} "
                        f"l_charb={row['l_charb']:.6f} l_freq={row['l_freq']:.6f} loss={row['loss']:.6f}"
                    )

                if iteration % args.val_freq == 0:
                    val_psnr, val_ssim = validate(model, val_loader, device)
                    best_psnr = max(best_psnr, val_psnr)
                    val_writer.writerow({"iter": iteration, "psnr": val_psnr, "ssim": val_ssim, "best_psnr": best_psnr})
                    val_f.flush()
                    tqdm.write(f"Validation iter={iteration} psnr={val_psnr:.4f} ssim={val_ssim:.4f} best={best_psnr:.4f}")

                if iteration % args.save_freq == 0 or iteration == args.total_iter:
                    state = {
                        "iter": iteration,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_psnr": best_psnr,
                        "args": vars(args),
                    }
                    torch.save(state, ckpt_dir / "latest.pth")
                    torch.save(state, ckpt_dir / f"iter_{iteration}.pth")

                progress.update(1)
                if iteration >= args.total_iter:
                    break
        progress.close()


if __name__ == "__main__":
    main()
