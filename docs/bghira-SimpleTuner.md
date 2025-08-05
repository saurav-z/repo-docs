<!--
SPDX-FileCopyrightText: 2024 SimpleTuner Contributors
SPDX-License-Identifier: MIT
-->

# SimpleTuner: Simplified AI Model Training 🚀

SimpleTuner empowers you to easily fine-tune and train various AI models with a focus on simplicity and understanding.  [View the original repo on GitHub](https://github.com/bghira/SimpleTuner).

*   **Privacy-Focused:** By default, no data is sent to third parties except through opt-in features like `report_to`, `push_to_hub`, or webhooks that require manual configuration.

## Key Features

*   **Simplified Training:** Designed for ease of use with good default settings.
*   **Versatile:** Supports diverse image quantities and aspect ratios.
*   **Cutting-Edge Techniques:** Integrates effective features while avoiding untested options.
*   **Multi-GPU Training:** Harness the power of multiple GPUs for faster training.
*   **Advanced Techniques:** Includes features like token-wise dropout (TREAD), image/video caching, and aspect bucketing.
*   **Model Support:** Comprehensive support for models including:
    *   Flux
    *   HiDream
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 2.0/2.1 & 3.0
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict

## Table of Contents

-   [Features](#key-features)
-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Hardware Requirements](#hardware-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Focus on intuitive configuration for ease of use.
*   **Versatility:** Supports a broad range of image and video datasets.
*   **Proven Features:** Employs only validated methods for effective training.

## Tutorial

Before starting, please review the [tutorial](/TUTORIAL.md) for important information.

*   **Quick Start:** Get up and running fast with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed Integration:** Optimize for memory-constrained systems using [DeepSpeed](/documentation/DEEPSPEED.md).
*   **Distributed Training:** Scale your training with the [Distributed Guide](/documentation/DISTRIBUTED.md) for multi-node setups.

## Hardware Requirements

The following are baseline hardware requirement suggestions. YMMV.

### NVIDIA

Generally, NVIDIA 3080 or higher is recommended.

### AMD

LoRA and full-rank tuning are verified to work on a 7900 XTX 24GB and MI300X. Requires more memory compared to equivalent Nvidia hardware due to a lack of `xformers`.

### Apple

LoRA and full-rank tuning tested on an M3 Max with 128GB memory, utilizing approximately **12GB** of "Wired" memory and **4GB** of system memory for SDXL. It is recommended to have a 24GB or greater machine for machine learning due to the lack of memory-efficient attention.

### HiDream [dev, full]

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)

### Flux.1 [dev, schnell]

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)
*   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
*   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### Auraflow

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

For information about SimpleTuner's associated toolkit, please see the [toolkit documentation](/toolkit/README.md).

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

*   **Enable Debug Logs:** Add `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file for detailed insights.
*   **Training Loop Analysis:** Use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` to analyze and troubleshoot the training loop.
*   **Configuration Options:** Consult [OPTIONS.md](/OPTIONS.md) for all available options.