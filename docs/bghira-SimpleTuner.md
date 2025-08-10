# SimpleTuner: Your Gateway to Simplified Diffusion Model Training 💹

**SimpleTuner empowers users to easily fine-tune cutting-edge diffusion models, accelerating your journey into AI image generation.** ([Original Repo](https://github.com/bghira/SimpleTuner))

> ℹ️ No data is sent to third parties except through opt-in flags like `report_to`, `push_to_hub`, or webhooks configured manually.

SimpleTuner prioritizes simplicity, making the complex process of training diffusion models accessible. This project is an evolving academic exercise, and contributions are welcome. Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group.

## Key Features

*   **Simplified Training:** Focus on ease of use with sensible defaults, reducing the need for extensive configuration.
*   **Versatile Support:** Handles a wide range of image and video datasets, from small collections to massive training sets.
*   **Cutting-Edge Techniques:** Integrates proven, advanced features for optimal performance and results.
*   **Multi-GPU Training:** Leverages multi-GPU setups for faster training.
*   **Token-wise Dropout:** Includes new techniques like TREAD for accelerated training of Wan 2.1/2.2 and Flux models.
*   **Caching:** Image, video, and caption data is cached to the hard drive for reduced memory consumption and faster training.
*   **Aspect Ratio Support:** Supports diverse image and video sizes and aspect ratios for flexible training.
*   **Model Support:** Full or LoRA/LyCORIS training for:
    *   HiDream
    *   Flux.1
    *   Wan Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict
    *   Qwen-Image
    *   Legacy Stable Diffusion (SD 1.x/2.x)
*   **DeepSpeed Integration:** Enables training of SDXL's full u-net with limited VRAM.
*   **Quantization Support:** Supports NF4/INT8/FP8 LoRA training to reduce VRAM consumption.
*   **EMA Weighting:** Optional Exponential Moving Average (EMA) for improved model stability.
*   **S3 Storage Support:** Train directly from S3-compatible storage providers (Cloudflare R2, Wasabi S3).
*   **ControlNet Training:** Supports ControlNet models for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts (MoE):** Support for training MoE models.
*   **Masked Loss Training:** Improves convergence and reduces overfitting.
*   **Prior Regularization:** Strong support for LyCORIS models.
*   **Webhook Integration:** Update Discord channels and other services with training progress.
*   **Hugging Face Hub Integration:** Easily upload and share your trained models.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#key-features)
*   [Hardware Requirements](#hardware-requirements)
*   [Scripts](#scripts)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizes sensible defaults to minimize configuration.
*   **Versatility:** Designed to handle diverse datasets.
*   **Cutting-Edge:** Focuses on proven and effective features.

## Tutorial

Refer to the [tutorial](/TUTORIAL.md) for detailed guidance.  Get a quick start with the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory-constrained systems, see the [DeepSpeed document](/documentation/DEEPSPEED.md).

For multi-node distributed training, [this guide](/documentation/DISTRIBUTED.md) is recommended.

## Hardware Requirements

### NVIDIA
*   General compatibility with NVIDIA GPUs, with the most up-to-date models recommended.

### AMD
*   LoRA and full-rank tuning verified working on a 7900 XTX 24GB and MI300X.
*   Will use more memory than Nvidia equivalent hardware due to lack of `xformers`.

### Apple
*   LoRA and full-rank tuning have been tested on an M3 Max with 128GB memory.
*   A 24GB or greater machine is recommended due to the lack of memory-efficient attention.

### HiDream
*   See original README for details.

### Flux.1
*   See original README for details.

### Auraflow
*   See original README for details.

### SDXL, 1024px
*   See original README for details.

### Stable Diffusion 2.x, 768px
*   16G or better recommended.

## Scripts

Details are in the original README.

## Toolkit

Find information about the associated toolkit in [the toolkit documentation](/toolkit/README.md).

## Setup

Follow the detailed setup instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.  For training loop analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.  Consult the [OPTIONS.md](/OPTIONS.md) documentation for more options.