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

Install a CUDA build of PyTorch first. On Vast.ai RTX 3090 hosts with NVIDIA driver 570 / CUDA 12.8, use the PyTorch CUDA 12.8 wheels:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Then install the project dependencies:

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

For experiment tracking, create a small fixed validation set from the clean pairs first:

```bash
python scripts/split_meta_info.py \
  --input meta_info/train_clean_psnr18.txt \
  --train-output meta_info/train_clean_psnr18_train.txt \
  --val-output meta_info/train_clean_psnr18_val.txt \
  --val-ratio 0.05 \
  --max-val 200

python scripts/make_val_crops.py \
  --data-root data/super-resolution-in-video-games \
  --meta-info meta_info/train_clean_psnr18_val.txt \
  --output-root data/val_crops \
  --crops-per-image 1
```

Download or provide a HAT-S SRx4 pretrained checkpoint, then generate a HAT training config:

```bash
python scripts/write_hat_config.py \
  --hat-root external/HAT \
  --data-root data/super-resolution-in-video-games \
  --pretrain /path/to/HAT-S_SRx4.pth \
  --meta-info meta_info/train_clean_psnr18_train.txt \
  --val-root data/val_crops \
  --val-freq 2000 \
  --total-iter 24000 \
  --output external/HAT/options/train/train_HAT-S_gamesr_baseline.yml
```

Then train:

```bash
cd external/HAT
python hat/train.py -opt options/train/train_HAT-S_gamesr_baseline.yml
```

This mirrors the notebook baseline: HAT-S x4, `gt_size=256`, batch size 4, hflip/rotation, plus an optional fixed validation crop set for experiment tracking.

With `--val-root`, HAT logs validation PSNR/SSIM every `--val-freq` iterations. The default validation metric uses RGB PSNR/SSIM with `crop_border=0`, which is closer to the Kaggle setup than the standard benchmark Y-channel PSNR.

## Training Logs

The HAT/BasicSR terminal log periodically includes:

- `epoch`, `iter`
- learning rate, shown as `lrs`
- iteration time and data loading time
- loss components from the model, for this baseline mainly `l_pix`
- validation metrics every `val_freq`, including `psnr`, `ssim`, and best-so-far values

For this L1-only HAT-S baseline, the key loss is `l_pix`. If later we add perceptual/GAN losses, BasicSR-style logs will also include components such as `l_g_percep`, `l_g_style`, `l_g_gan`, `l_d_real`, and `l_d_fake`.

Logs and TensorBoard files are written under `external/HAT/experiments/<experiment_name>/` and `external/HAT/tb_logger/<experiment_name>/`.

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
- Loss: L1 loss from the official HAT config, logged as `l_pix`
- Metric: Kaggle PSNR; local validation uses fixed `64x64 -> 256x256` RGB crops

Recommended next experiments are listed in [docs/baseline_notes.md](docs/baseline_notes.md).
