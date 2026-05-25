# Literature Notes

These notes summarize the user-provided PDFs in `/Users/ed/Downloads` and connect them to the Kaggle baseline.

## HAT

Source: `HAT.pdf`

HAT stands for Hybrid Attention Transformer. It combines window-based self-attention, channel attention, and overlapping cross-attention. The main motivation is that previous SR Transformers such as SwinIR model local detail well but still do not use enough broader spatial context. HAT is a strong fit for this competition because the Kaggle metric is PSNR and HAT is designed for high-fidelity image restoration.

Baseline relevance:

- Directly matches the 3rd-place code.
- HAT-S is small enough to fine-tune on Kaggle-style GPUs.
- Same-task pretraining is important, so starting from SRx4 weights is better than training from scratch.

## GameSR

Source: `GameSR.pdf`

GameSR targets real-time, engine-independent super-resolution for interactive gaming and cloud gaming. It works on rendered or encoded frames instead of requiring game engine buffers such as motion vectors or depth maps. The paper emphasizes lightweight modules, PixelUnshuffle, reparameterized convolutional blocks, and ConvLSTM for temporal information.

Baseline relevance:

- The Kaggle task is image SR, not video SR, but the domain is game content.
- GameSR is useful for improvement ideas after reproducing HAT: lightweight CNN blocks, reparameterization, and game-specific inductive bias.
- Temporal ConvLSTM is not directly usable unless we have frame sequences, so it is a later-stage idea, not the first baseline.

## NTIRE 2025 RAW Restoration and Super-Resolution

Source: `NTIRE 2025 Challenge on RAW Image Restoration and Super-Resolution.pdf`

This challenge report surveys strong RAW restoration/SR methods. It stresses degradation modeling, blur/noise robustness, efficient models, and evaluation by PSNR/SSIM. Although RAW SR is different from RGB game screenshots, the experimental style is useful: clear baselines, parameter counts, ablations, and fidelity metrics.

Baseline relevance:

- Use PSNR-centered loss choices and report PSNR cleanly.
- Consider efficient-model ideas only after HAT-S baseline is stable.
- Distillation and reparameterization are plausible improvement directions if runtime/model size becomes part of the presentation.

## RBSFormer

Source: `RBSFomer.pdf`

RBSFormer is an enhanced Transformer for RAW image SR. It uses Enhanced Cross-Covariance Attention and Enhanced Gated Feed-forward Network, with Charbonnier loss and ensemble strategies. Its key takeaway is not the RAW-specific input, but the pattern of improving a Transformer restoration model through attention/feed-forward design, robust loss, and ensemble.

Baseline relevance:

- Charbonnier loss is a reasonable ablation against HAT's default L1 loss.
- Ensemble and multi-configuration strategies can inspire final submissions.
- RAW-specific Bayer assumptions do not transfer to RGB game screenshots.

Implementation note:

- The repository's `RBSFormerRGB` is an RGB adaptation, not the original RAW challenge model.
- It preserves EXCA, EGFN, inception depthwise projection, Charbonnier loss, and frequency loss.
- The RGB version predicts residual corrections over bicubic upsampling by default, which is a PSNR-oriented adaptation for this competition.
- Because no public official implementation was found during setup, use it as an experimental method rather than a strict paper reproduction.

## Suggested Story for the Final Project

Start with a faithful HAT-S reproduction. Then show that the largest early gain comes from data quality: mismatched pair filtering and duplicate-aware validation. After that, test training and inference improvements such as loss choice, longer fine-tuning, TTA, and checkpoint averaging. This gives the project a clean arc from reproduction to measurable improvement.
