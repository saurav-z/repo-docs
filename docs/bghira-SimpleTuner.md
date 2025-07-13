# SimpleTuner: Fine-Tune Your Diffusion Models with Ease

**Unlock the power of custom diffusion models with SimpleTuner, but remember to back up your training data!**

SimpleTuner simplifies the process of fine-tuning diffusion models, focusing on ease of use and accessibility for researchers and enthusiasts.  This repository is a community effort, and contributions are welcome.  Find us and ask questions on [Discord](https://discord.com/invite/eq3cAMZtCC) via Terminus Research Group.

🔗  [View the original repository on GitHub](https://github.com/bghira/SimpleTuner)

## Key Features

*   **Simplified Fine-Tuning:** Designed for easy understanding and use, with sensible default settings.
*   **Versatile:** Handles a wide range of datasets, from small to extremely large.
*   **Cutting-Edge Technology:** Integrates proven features for optimal results.
*   **Multi-GPU Training:** Accelerate training with multiple GPUs.
*   **Memory Optimization:** Image, video, and caption caching reduces VRAM consumption and speeds up training.
*   **Aspect Bucketing:** Supports diverse image/video sizes and aspect ratios.
*   **LoRA/Full U-Net Training:** Supports LoRA or full U-Net training for SDXL.
*   **DeepSpeed Integration:** Allows for training large models even with limited VRAM.
*   **Quantization Support:** Train with lower precision (NF4/INT8/FP8) to reduce VRAM usage.
*   **S3 Training:** Train directly from cloud storage (e.g., Cloudflare R2, Wasabi S3).
*   **ControlNet Training:** Train ControlNet models for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts (MoE) Training:** Experiment with MoE models for lightweight diffusion models.
*   **Masked Loss Training:** Enhance convergence and reduce overfitting.
*   **Hugging Face Hub Integration:** Easily upload and share your models.
*   **Comprehensive Support:** Support for HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, and legacy Stable Diffusion models.

## Table of Contents

- [Design Philosophy](#design-philosophy)
- [Tutorial](#tutorial)
- [Features](#features)
    - [HiDream](#hidream)
    - [Flux.1](#flux1)
    - [Wan Video](#wan-video)
    - [LTX Video](#ltx-video)
    - [PixArt Sigma](#pixart-sigma)
    - [NVLabs Sana](#nvlabs-sana)
    - [Stable Diffusion 3](#stable-diffusion-3)
    - [Kwai Kolors](#kwai-kolors)
    - [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)
- [Hardware Requirements](#hardware-requirements)
- [Toolkit](#toolkit)
- [Setup](#setup)
- [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizing easy-to-use settings for common tasks.
*   **Versatility:** Adaptability to diverse dataset sizes and image/video formats.
*   **Innovation:** Only incorporating tested and effective features.

## Tutorial

Get started by exploring this README and then diving into the [tutorial](/TUTORIAL.md).  For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.  For memory-constrained systems, explore [DeepSpeed documentation](/documentation/DEEPSPEED.md). For multi-node training, see [this guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

Consult the [Hardware Requirements](#hardware-requirements) section for detailed information.

## Toolkit

Learn more about the included toolkit [here](/toolkit/README.md).

## Setup

Follow the detailed setup instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logging for detailed information by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG`.  For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. Consult [this documentation](/OPTIONS.md) for a full list of options.