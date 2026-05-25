# Baseline Notes

## What the 3rd-place notebook does

The notebook `3rd_code.ipynb` is a practical Kaggle workflow around the official HAT repository. Its important ideas are:

- Use HAT-S SRx4 as the base model.
- Fine-tune on the competition LR/HR pairs.
- Use paired random crop because the public test LR images are cropped to `64x64` and the hidden HR targets are cropped to `256x256`.
- Use horizontal flip and rotation augmentation.
- Detect dirty training pairs. The notebook says some pairs do not contain the same content, and many pairs are duplicates.
- Filter dirty pairs by computing PSNR between resized LR and HR images. The notebook uses threshold `18`.
- Generate the Kaggle submission with the required lossless RLE + zlib + base64 encoding.

## Why we changed the notebook workflow

The notebook modifies HAT's `train.py` so bad pairs are skipped during training. This repo instead creates a clean `meta_info_file` before training. That is easier to reproduce and easier to compare in ablation studies.

The baseline remains faithful to the original idea:

- Same HAT-S x4 family.
- Same crop size.
- Same augmentation.
- Same PSNR threshold.
- Same Kaggle encoding.

## Baseline Commands

```bash
python scripts/download_data.py --link data/super-resolution-in-video-games

python scripts/build_clean_meta.py \
  --data-root data/super-resolution-in-video-games \
  --threshold 18 \
  --output meta_info/train_clean_psnr18.txt \
  --rejects meta_info/rejected_psnr18.csv \
  --summary meta_info/clean_psnr18_summary.json

python scripts/write_hat_config.py \
  --hat-root external/HAT \
  --data-root data/super-resolution-in-video-games \
  --pretrain /path/to/HAT-S_SRx4.pth \
  --meta-info meta_info/train_clean_psnr18.txt \
  --total-iter 24000 \
  --output external/HAT/options/train/train_HAT-S_gamesr_baseline.yml

cd external/HAT
python hat/train.py -opt options/train/train_HAT-S_gamesr_baseline.yml
```

## Improvement Plan After Baseline

1. Add a validation split that avoids duplicate leakage.
2. Sweep cleaning threshold: no cleaning, `16`, `18`, `20`.
3. Compare L1, MSE, and Charbonnier loss.
4. Try longer fine-tuning schedules from the same checkpoint.
5. Add inference TTA.
6. Average checkpoints or outputs if local validation and Kaggle public score agree.
7. Compare HAT-S with HAT, SwinIR, or a lightweight GameSR-inspired CNN only after the HAT-S baseline is stable.

## Reporting Checklist

- Kaggle score screenshot.
- Baseline method diagram.
- Data cleaning examples and threshold histogram.
- Ablation table.
- Visual comparisons on representative crops.
- Team contribution table required by the course PDF.

