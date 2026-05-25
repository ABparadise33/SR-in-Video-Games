# SR in Video Games

Baseline reproduction for the Kaggle competition [Super Resolution in Video Games](https://www.kaggle.com/competitions/super-resolution-in-video-games/data).

The first target is to reproduce the public 3rd-place style solution from `3rd_code.ipynb`: HAT-S x4 fine-tuning, paired random crop, flip/rotation augmentation, and PSNR-based filtering for mismatched LR/HR training pairs. After this baseline is stable, use it as the reference point for ablations and improvements.

## Repository Layout

```text
.
├── 3rd_code.ipynb
├── configs/
│   └── baseline_hat_s_gamesr.yaml
├── docs/
│   ├── baseline_notes.md
│   └── literature_notes.md
├── scripts/
│   ├── build_clean_meta.py
│   ├── download_data.py
│   ├── make_submission_hat.py
│   ├── setup_hat.py
│   └── write_hat_config.py
└── requirements.txt
```

Generated folders such as `data/`, `external/HAT/`, `meta_info/`, `submissions/`, and model checkpoints are ignored by git because they are large or reproducible.

## 1. Install Project Dependencies

```bash
pip install -r requirements.txt
```

Then set up HAT:

```bash
python scripts/setup_hat.py --hat-root external/HAT --install
```

If HAT was already cloned, the script will reuse it.

If training later fails with `torchvision.transforms.functional_tensor`, run:

```bash
python scripts/setup_hat.py --hat-root external/HAT --fix-basicsr-torchvision
```

## 2. Download Kaggle Data

The dataset is large, around 51 GB. This script uses the same KaggleHub API style you provided:

```bash
python scripts/download_data.py --link data/super-resolution-in-video-games
```

It prints the KaggleHub cache path and creates a local symlink at `data/super-resolution-in-video-games`.

Expected competition structure:

```text
data/super-resolution-in-video-games/
├── sample_submission.csv
├── train/
│   ├── hr/
│   └── lr/
└── test/
    └── lr/
```

## 3. Build Clean Training Metadata

The 3rd-place notebook notes that some LR/HR pairs are mismatched and skips pairs whose resized pair PSNR is below 18. This repo makes that step explicit and reproducible:

```bash
python scripts/build_clean_meta.py \
  --data-root data/super-resolution-in-video-games \
  --threshold 18 \
  --output meta_info/train_clean_psnr18.txt \
  --rejects meta_info/rejected_psnr18.csv \
  --summary meta_info/clean_psnr18_summary.json
```

The output `txt` file is a BasicSR/HAT-compatible `meta_info_file`.

## 4. Write HAT Baseline Config

Download or provide a HAT-S SRx4 pretrained checkpoint, then generate a HAT training config:

```bash
python scripts/write_hat_config.py \
  --hat-root external/HAT \
  --data-root data/super-resolution-in-video-games \
  --pretrain /path/to/HAT-S_SRx4.pth \
  --meta-info meta_info/train_clean_psnr18.txt \
  --total-iter 24000 \
  --output external/HAT/options/train/train_HAT-S_gamesr_baseline.yml
```

Then train:

```bash
cd external/HAT
python hat/train.py -opt options/train/train_HAT-S_gamesr_baseline.yml
```

This mirrors the notebook baseline: HAT-S x4, `gt_size=256`, batch size 4, hflip/rotation, no validation set by default.

## 5. Create a Kaggle Submission

```bash
python scripts/make_submission_hat.py \
  --hat-root external/HAT \
  --data-root data/super-resolution-in-video-games \
  --checkpoint external/HAT/experiments/train_HAT-S_gamesr_baseline/models/net_g_latest.pth \
  --output submissions/hat_s_baseline.csv
```

Optional TTA:

```bash
python scripts/make_submission_hat.py \
  --hat-root external/HAT \
  --data-root data/super-resolution-in-video-games \
  --checkpoint /path/to/checkpoint.pth \
  --output submissions/hat_s_tta.csv \
  --tta
```

## Baseline Definition

Use this as the initial score anchor:

- Model: HAT-S x4
- Pretraining: HAT-S SRx4 checkpoint
- Fine-tuning data: Kaggle train LR/HR pairs
- Crop: paired random crop with `gt_size=256`
- Augmentation: horizontal flip and rotation
- Cleaning: remove or skip LR/HR pairs with pair PSNR `< 18`
- Loss: L1 loss from the official HAT config
- Metric: Kaggle PSNR

Recommended next experiments are listed in [docs/baseline_notes.md](docs/baseline_notes.md).
