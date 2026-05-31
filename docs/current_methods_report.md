# HAT and RBSFormer Method Report

本文件整理目前 repository 中已實作與已測試的兩條 Super Resolution in Video Games 方法線：HAT 與 RGB-adapted RBSFormer。內容可作為期末報告 Method、Experiment、Ablation 與 Discussion 的基礎稿。

## 1. Task Setting

Kaggle Super Resolution in Video Games 的目標是從低解析度遊戲畫面重建高解析度影像。資料格式為 paired image super-resolution：

- Training input: `train/lr/*.png`
- Training target: `train/hr/*.png`
- Test input: `test/lr/*.png`
- Scale factor: `x4`
- Test LR crop size: `64 x 64`
- Test HR output size: `256 x 256`
- Main metric: Kaggle PSNR
- Submission format: each predicted HR image must be encoded as RLE + zlib + base64 bytes representation

目前 pipeline 的共同資料處理流程如下：

1. 下載 Kaggle competition data。
2. 建立 clean meta file，過濾 LR/HR 內容不一致的 pair。
3. 從 clean training pairs 切出固定 validation crops。
4. 使用 paired random crop 進行訓練。
5. 使用 RGB PSNR/SSIM 追蹤 local validation。
6. 對 test/lr 進行推論，輸出符合 Kaggle 格式的 submission CSV。

## 2. Data Cleaning and Validation Design

第三名 notebook 提到資料中存在 mismatched LR/HR pairs。因此目前 repository 先建立 clean meta file，而不是在 dataloader 中動態跳過壞資料。

### 2.1 Cleaning Rule

使用 `scripts/build_clean_meta.py`：

- 將 LR 與 HR resize 到共同檢查尺寸。
- 計算 RGB PSNR。
- 若 pair PSNR `< 18`，視為內容不一致或品質異常，排除。
- 產出 BasicSR/HAT 可讀的 `meta_info_file`。

目前 baseline 使用：

```text
meta_info/train_clean_psnr18_train.txt
meta_info/train_clean_psnr18_val.txt
```

### 2.2 Validation Crops

使用 `scripts/make_val_crops.py` 從 validation split 建立固定 crop：

- LR crop: `64 x 64`
- HR crop: `256 x 256`
- 與 Kaggle test crop size 對齊
- 每 `val_freq` iterations 計算 PSNR/SSIM

這個 local validation 不等同 Kaggle hidden test set，但可用於觀察訓練趨勢與比較相同 setting 下的模型。

## 3. HAT Method

### 3.1 Model Motivation

HAT, Hybrid Attention Transformer, 是一個高品質 image restoration / super-resolution Transformer。相較於只使用 window self-attention 的模型，HAT 結合：

- Window-based self-attention
- Channel attention
- Overlapping cross-attention
- Residual hybrid attention groups

這使它能同時處理局部紋理與較大範圍的空間關係。遊戲畫面通常包含邊緣清楚的物件、UI、文字、材質紋理與規則幾何結構，因此 HAT 這類 fidelity-oriented SR 模型適合 PSNR-driven 的競賽。

### 3.2 Current HAT-S Baseline

目前重現第三名 code 的主要 baseline 是 HAT-S SRx4 fine-tuning。

Network config:

```text
model: HAT-S
scale: 4
input channels: 3
window_size: 16
depths: [6, 6, 6, 6, 6, 6]
embed_dim: 144
num_heads: [6, 6, 6, 6, 6, 6]
compress_ratio: 24
squeeze_factor: 24
upsampler: pixelshuffle
resi_connection: 1conv
```

Training config:

```text
pretrain: HAT-S_SRx4.pth
gt_size: 256
lr crop size: 64
batch size: 4
optimizer: Adam
learning rate: 1e-4
betas: [0.9, 0.99]
weight decay: 0
scheduler: MultiStepLR from official HAT config
loss: L1 pixel loss
logged loss: l_pix
augmentation: horizontal flip + rotation
EMA: 0.999
validation frequency: 2000 iterations
```

The model is trained through the official HAT/BasicSR framework, with repository helpers generating the YAML config and parsing logs into CSV.

### 3.3 HAT Training and Logging

Relevant scripts:

- `scripts/write_hat_config.py`: generate HAT YAML from an official template.
- `scripts/train_hat_with_csv.py`: run HAT training and save terminal log + CSV metrics.
- `scripts/parse_hat_log_to_csv.py`: parse existing HAT terminal logs into CSV.
- `scripts/plot_training_curves.py`: plot loss and validation curves.

HAT baseline logs:

- `l_pix`: L1 pixel loss
- `psnr`: validation PSNR
- `ssim`: validation SSIM
- `best_psnr`, `best_ssim`: best validation metrics so far

Because current HAT training uses only L1 loss, there are no frequency/perceptual/GAN loss components in HAT logs. If perceptual or GAN training is added later, BasicSR-style logs may include components such as `l_g_percep`, `l_g_style`, `l_g_gan`, `l_d_real`, and `l_d_fake`.

### 3.4 HAT Inference and TTA

Inference uses `scripts/make_submission_hat.py`.

The script:

1. Loads the HAT architecture.
2. Loads checkpoint weights, preferring `params_ema` when available.
3. Reads test LR images as RGB.
4. Predicts HR images.
5. Converts prediction back to BGR before Kaggle encoding.
6. Encodes each image with flatten -> RLE -> zlib -> base64.

TTA uses x8 geometric self-ensemble:

- identity
- horizontal flip
- vertical flip
- horizontal + vertical flip
- transpose
- transpose + horizontal flip
- transpose + vertical flip
- transpose + both flips

Predictions are inverted back to the original orientation and averaged. TTA improved the reliability of submission scores and is used for final HAT submissions.

### 3.5 HAT Results So Far

| Method | Local Val PSNR | Local Val SSIM | Kaggle Private | Kaggle Public | Notes |
|---|---:|---:|---:|---:|---|
| HAT-S 24k, fixed BGR, TTA | 28.3656 | 0.7904 | 33.440 | 33.425 | Reproduced baseline, correct submission encoding |
| HAT-S continued to 60k, TTA | 28.6214 | 0.7952 | 33.614 | 33.592 | Best current HAT-S result |

Observed trend:

- HAT-S 24k to 60k improves local validation PSNR from `28.3656` to `28.6214`.
- Kaggle private score improves from `33.440` to `33.614`.
- The training loss `l_pix` is noisy batch-by-batch, but rolling averages still decrease slightly.
- Current HAT-S remains below the project target `34.476`, so a stronger pretrained backbone is needed.

### 3.6 Planned HAT Full / HAT-L Extension

To pass `34.x`, the next main experiment should move from HAT-S to HAT full or HAT-L.

HAT full SRx4 ImageNet-pretrain config:

```text
model: HAT
scale: 4
depths: [6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6]
compress_ratio: 3
squeeze_factor: 30
pretrain: HAT_SRx4_ImageNet-pretrain.pth
recommended lr: 1e-5
recommended batch size on RTX 3090: 1 or 2
```

HAT-L SRx4 ImageNet-pretrain config:

```text
model: HAT-L
scale: 4
depths: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
embed_dim: 180
num_heads: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
compress_ratio: 3
squeeze_factor: 30
pretrain: HAT-L_SRx4_ImageNet-pretrain.pth
recommended batch size on RTX 3090: 1
```

HAT full is the next recommended experiment. HAT-L should be tested only after confirming that HAT full improves local validation because HAT-L is much slower and more memory intensive.

## 4. RGB-Adapted RBSFormer Method

### 4.1 Original RBSFormer Motivation

RBSFormer was proposed for RAW image restoration and super-resolution. The original problem assumes RAW Bayer input and focuses on RAW-specific restoration. The current repository does not reproduce the RAW setting directly. Instead, it adapts the main architectural ideas to 3-channel RGB game screenshots.

The motivation for including RBSFormer is experimental comparison:

- It provides a non-HAT Transformer-like baseline.
- It uses Charbonnier loss and optional frequency loss.
- It provides a useful contrast between pretrained HAT and a smaller custom model trained mostly from scratch.

### 4.2 Current RBSFormerRGB Architecture

Implemented in `srvg/models/rbsformer_rgb.py`.

Top-level structure:

```text
RGB LR input
  -> 3x3 conv_first
  -> sequence of TransformerBlock
  -> 3x3 conv_body
  -> skip with shallow feature
  -> conv + PixelShuffle x4
  -> optional bicubic residual upsample
  -> RGB HR output
```

Default network config:

```text
channels: 60
num_blocks: 8
num_heads: 6
ffn_expansion: 2.0
scale: 4
upsampler: PixelShuffle
optional residual upsample: bicubic LR x4 + predicted residual
```

Each TransformerBlock contains:

1. EXCA: Enhanced Cross-Covariance Attention
2. EGFN: Enhanced Gated Feed-Forward Network

### 4.3 EXCA: Enhanced Cross-Covariance Attention

EXCA is a channel-covariance style attention adapted from RBSFormer.

Implementation details:

- Apply `LayerNorm2d`.
- Generate Q, K, V with `Project`.
- `Project` uses `1x1 conv + InceptionDWConv`.
- Q and K are normalized.
- Attention is computed through cross-covariance over channel groups.
- A learnable temperature is used per head.
- The attention output is projected and added back through a residual connection.

Compared with spatial self-attention, EXCA is cheaper for image restoration because it computes attention over channel dimensions rather than all spatial tokens.

### 4.4 InceptionDWConv

`InceptionDWConv` splits channels into four groups:

- group 1: depthwise `3x3`
- group 2: depthwise `1x11`
- group 3: depthwise `11x1`
- group 4: identity

This provides anisotropic large-kernel context without the full cost of dense large convolutions.

### 4.5 EGFN: Enhanced Gated Feed-Forward Network

EGFN is the feed-forward branch:

1. Apply `LayerNorm2d`.
2. Use `1x1 conv` to expand channels to `2 * hidden`.
3. Apply `InceptionDWConv`.
4. Split into two tensors.
5. Use gated activation: `GELU(y1) * y2`.
6. Project back to the original channel dimension.
7. Add residual connection.

This is similar in spirit to gated feed-forward blocks used in image restoration Transformers, where multiplicative gating helps preserve useful high-frequency details.

### 4.6 RBSFormer Training

Implemented in `scripts/train_rbsformer_rgb.py`.

Dataset:

- Uses the same clean meta file as HAT.
- Random LR crop size: `64 x 64`.
- Corresponding HR crop size: `256 x 256`.
- Augmentation: horizontal flip, vertical flip, random rotation.

Optimizer and scheduler:

```text
optimizer: AdamW
initial lr: 7e-4
betas: [0.9, 0.999]
weight_decay: 0
scheduler: CosineAnnealingLR
eta_min: lr * 0.01
gradient clipping: 1.0
```

Loss:

```text
l_charb = CharbonnierLoss(pred, gt)
l_freq = mean absolute FFT difference
loss = l_charb + freq_loss_weight * l_freq
```

The currently logged CSV contains:

- `iter`
- `lr`
- `l_charb`
- `l_freq`
- `loss`

Validation contains:

- `iter`
- `psnr`
- `ssim`
- `best_psnr`

### 4.7 RBSFormer Results So Far

| Method | Local Val PSNR | Local Val SSIM | Kaggle Private | Kaggle Public | Notes |
|---|---:|---:|---:|---:|---|
| RBSFormerRGB 100k, TTA | 28.1830 | 0.8883 | 33.387 | 33.376 | Best current RBSFormer submission |

Observed trend:

- RBSFormer improves steadily through 100k iterations.
- Local PSNR reaches `28.1830`, below HAT-S 60k's `28.6214`.
- Kaggle score is also below HAT-S 60k.
- SSIM is numerically high, but this SSIM implementation differs from HAT/BasicSR validation, so it should not be directly compared with HAT SSIM.

Interpretation:

- RBSFormerRGB is useful as an ablation model and architectural comparison.
- It is not currently the main score-improving path.
- Its biggest disadvantage is lack of strong SRx4 pretrained weights for this exact RGB game-image task.
- HAT benefits heavily from pretrained SR weights, while RBSFormerRGB is closer to training a custom architecture from scratch.

## 5. HAT vs RBSFormer Comparison

| Aspect | HAT-S / HAT | RBSFormerRGB |
|---|---|---|
| Original domain | RGB image SR | RAW image SR |
| Current implementation | Official HAT/BasicSR | Custom RGB adaptation |
| Pretraining | Strong SRx4 pretrained checkpoints | No equivalent RGB game SR pretraining |
| Main loss | L1 pixel loss | Charbonnier + optional frequency loss |
| Optimizer | Adam | AdamW |
| Scheduler | MultiStepLR | CosineAnnealingLR |
| Inference | TTA supported | TTA supported |
| Best current Kaggle private | 33.614 | 33.387 |
| Role in project | Main performance path | Comparison/ablation method |

Main lesson:

For this competition, architecture matters, but pretraining matters more. HAT-S with pretrained SR weights outperforms RBSFormerRGB despite RBSFormer using Charbonnier/frequency losses, because the HAT checkpoint already contains strong natural-image SR priors.

## 6. Current Best Score Summary

| Submission | Private | Public | Comment |
|---|---:|---:|---|
| HAT-S 24k fixed BGR TTA | 33.440 | 33.425 | Correct HAT-S baseline |
| RBSFormerRGB 100k TTA | 33.387 | 33.376 | Custom RGB RBSFormer |
| HAT-S continued 60k TTA | 33.614 | 33.592 | Current best |

Target for full model performance:

```text
33.8 * 1.02 = 34.476
```

Current gap:

```text
34.476 - 33.614 = 0.862
```

This gap is large enough that small tricks alone are unlikely to be sufficient. The next step should be model capacity and pretraining upgrade.

## 7. Recommended Next Experiments

### 7.1 High Priority

1. Fine-tune HAT full from `HAT_SRx4_ImageNet-pretrain.pth`.
2. Use TTA for every serious submission.
3. Track local validation and Kaggle public/private score together.
4. Save checkpoints every 10k or 20k iterations and submit promising checkpoints.

### 7.2 Medium Priority

1. Test HAT-L if HAT full clearly improves over HAT-S.
2. Try stricter cleaning threshold such as PSNR `20` or `22`.
3. Try output ensemble:
   - HAT-S 60k
   - HAT full best checkpoint
   - possibly RBSFormerRGB
4. Try checkpoint averaging for HAT full if validation curve becomes noisy.

### 7.3 Lower Priority

1. Add frequency loss to HAT.
2. Add perceptual/GAN loss.
3. Train a CNN or GAN-based method for performance comparison.

Reason: Kaggle metric is PSNR. GAN/perceptual losses may improve visual sharpness but often reduce PSNR, so they are better for discussion or visual comparison than for leaderboard improvement.

## 8. Suggested Report Narrative

The final report can be organized as:

1. Problem definition: x4 SR for video game images.
2. Baseline reproduction: third-place HAT-S method.
3. Data cleaning: remove mismatched LR/HR pairs by PSNR threshold.
4. HAT method: pretrained Transformer SR model with L1 loss and TTA.
5. RBSFormer method: RGB adaptation with EXCA, EGFN, Charbonnier and frequency loss.
6. Experimental comparison:
   - HAT-S 24k
   - HAT-S 60k
   - RBSFormerRGB 100k
   - HAT full / HAT-L if completed
7. Ablation:
   - TTA on/off
   - training length
   - model architecture
   - loss type
   - data cleaning threshold
8. Discussion:
   - pretrained HAT currently dominates
   - RBSFormer shows alternative architecture but lacks pretrained advantage
   - achieving 34.x likely requires HAT full/HAT-L or ensemble

## 9. References

- HAT: Hybrid Attention Transformer for Image Restoration.
- RBSFormer: Enhanced Transformer for RAW Image Super-Resolution.
- GameSR: Engine-independent super-resolution for video games.
- NTIRE 2025 RAW Restoration and Super-Resolution challenge report.
- Kaggle Super Resolution in Video Games competition.
