# SimpleTuner 💹: Your gateway to simplified AI model training.

[SimpleTuner](https://github.com/bghira/SimpleTuner) empowers you to fine-tune various diffusion models with ease, offering a user-friendly experience for both beginners and experienced users. 

> ℹ️  This project prioritizes user privacy; no data is sent to third parties unless explicitly enabled via `report_to`, `push_to_hub`, or manually configured webhooks.

**Key Features:**

*   **Simplicity:** Easy-to-understand codebase with sensible default settings.
*   **Versatility:** Supports a wide range of image quantities and aspect ratios.
*   **Cutting-Edge:** Integrates proven features for optimal performance.
*   **Multi-GPU Training:**  Leverage the power of multiple GPUs for faster training.
*   **Accelerated Training:**  Utilizes token-wise dropout techniques (TREAD) for speed.
*   **Caching:** Caches image, video, and caption features to accelerate training and reduce memory usage.
*   **Aspect Bucketing:** Train with diverse image/video sizes and aspect ratios.
*   **Broad Model Support:** Train LoRA, LyCORIS, and full U-Net models, including support for:
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
*   **Memory Optimization:** Features DeepSpeed integration and quantization for training on resource-constrained hardware.
*   **S3 Training:** Train directly from S3-compatible storage providers.
*   **ControlNet Support:**  Train ControlNet models for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts:** Train lightweight, high-quality diffusion models.
*   **Advanced Training Techniques:**  Includes masked loss training, prior regularization, and webhook support for progress updates.
*   **Hugging Face Hub Integration:** Seamlessly upload models and generate model cards.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Wan Video](#wan-video)
    -   [LTX Video](#ltx-video)
    -   [PixArt Sigma](#pixart-sigma)
    -   [NVLabs Sana](#nvlabs-sana)
    -   [Stable Diffusion 3](#stable-diffusion-3)
    -   [Kwai Kolors](#kwai-kolors)
    -   [Lumina2](#lumina2)
    -   [Cosmos2 Predict](#cosmos2-predict)
    -   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)
-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream](#hidream-dev-full)
    -   [Flux.1](#flux1-dev-schnell)
    -   [Auraflow](#auraflow)
    -   [SDXL, 1024px](#sdxl-1024px)
    -   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity:** Emphasizes easy-to-use default settings for a streamlined experience.
-   **Versatility:** Designed to handle diverse datasets, from small to large.
-   **Cutting-Edge:**  Focuses on integrating effective features.

## Tutorial

For a comprehensive guide, start with the [main tutorial](/TUTORIAL.md). The [Quick Start](/documentation/QUICKSTART.md) provides a faster path to get up and running. Explore [DeepSpeed document](/documentation/DEEPSPEED.md) for optimizing memory usage. For multi-node training configuration guidance, consult [this guide](/documentation/DISTRIBUTED.md)

## Hardware Requirements

*(See detailed requirements below for each model type.)*

### NVIDIA

*(General guidance: 3080 and up recommended.)*

### AMD

*(Verified LoRA and full-rank tuning on 7900 XTX 24GB and MI300X. May require more memory due to the lack of `xformers`.)*

### Apple

*(Verified LoRA and full-rank tuning on M3 Max with 128GB (requires ~12G "Wired" memory + ~4G system memory for SDXL).  Consider a 24GB+ machine for ML on M-series hardware due to memory-efficient attention limitations.)*

### HiDream [dev, full]

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)

### Flux.1 [dev, schnell]

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)
-   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
-   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### Auraflow

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)
-   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
-   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

-   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
-   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
-   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
-   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
-   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

-   16G or better

---

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md) for details on the included tools.

## Setup

See the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For training loop performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

Consult [this documentation](/OPTIONS.md) for a complete list of options.