# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner empowers you to effortlessly train and fine-tune cutting-edge diffusion models, offering a user-friendly approach for both beginners and experienced users.** Check out the [original repository](https://github.com/bghira/SimpleTuner) for the source code and more details.

## Key Features:

*   **Simplicity-Focused:** Designed for ease of use with sensible default settings to minimize tinkering.
*   **Versatile Training:** Supports a wide range of image and video datasets, from small to large.
*   **Cutting-Edge Capabilities:** Integrates advanced features that have proven efficacy.
*   **Multi-GPU Training:** Efficiently utilizes multiple GPUs for faster training.
*   **Caching for Speed:** Caches images, videos, and captions for faster training with reduced memory usage.
*   **Aspect Ratio Support:** Supports a variety of image/video sizes and aspect ratios, enabling widescreen and portrait training.
*   **LoRA/LyCORIS Support:** Offers LoRA and LyCORIS training for various models including SDXL, SD3, and others, reducing VRAM consumption.
*   **DeepSpeed Integration:** Integrates with DeepSpeed for training large models on systems with limited VRAM.
*   **Quantization:** Supports quantized NF4/INT8/FP8 LoRA training to reduce VRAM usage.
*   **EMA for Stability:** Includes optional Exponential Moving Average (EMA) for improved training stability.
*   **S3 Training:** Train directly from S3-compatible storage providers.
*   **ControlNet Training:** Supports ControlNet training for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts:** Supports Mixture of Experts training for lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Supports masked loss training for superior convergence and reduced overfitting.
*   **Prior Regularization:** Strong prior regularization training support for LyCORIS models
*   **Webhook Support:** Supports webhooks for real-time training progress updates (e.g., to Discord).
*   **Hugging Face Hub Integration:** Seamless model upload and model card generation to the Hugging Face Hub.
*   **Model Support:** Supports a wide range of models including:
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

## Table of Contents

-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream](#hidream-dev-full)
    -   [Flux.1](#flux1-dev-schnell)
    -   [SDXL, 1024px](#sdxl-1024px)
    -   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Hardware Requirements

### NVIDIA

Any modern NVIDIA GPU (3080 and up) should work well.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.
Lacking `xformers`, it will use more memory than Nvidia equivalent hardware.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.

### HiDream \[dev, full]

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)

### Flux.1 \[dev, schnell]

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)
*   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
*   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

*   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
*   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
*   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
*   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
*   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

*   16G or better

## Toolkit

For information about the toolkit, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`).
For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
For all options, see [this documentation](/OPTIONS.md).