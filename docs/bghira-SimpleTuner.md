# SimpleTuner: Easy-to-Use Training for Diffusion Models 🚀

**SimpleTuner simplifies diffusion model training, providing an accessible experience for both beginners and experts, while maintaining a focus on performance and ease of use.  [Explore the SimpleTuner repository](https://github.com/bghira/SimpleTuner) for a deeper dive.**

> ⚠️ **Warning**: Back up your training data. This repository's scripts have the potential to damage your training data. Always maintain backups before proceeding.

SimpleTuner offers a streamlined approach to training diffusion models, emphasizing simplicity and a focus on easily understood code.  This project is a collaborative effort, and contributions are welcome.

Join our community and ask questions on [Discord](https://discord.com/invite/eq3cAMZtCC) via Terminus Research Group.

## Key Features

*   **Versatile Model Support:** Train a wide range of diffusion models, including:
    *   Flux.1
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 2.0/2.1
    *   Stable Diffusion 3.0
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   HiDream
    *   Auraflow

*   **Memory Optimization:**
    *   Image, video, and caption caching for faster and more efficient training.
    *   DeepSpeed integration for training on resource-constrained systems.
    *   Quantization options (NF4/INT8/FP8) to reduce VRAM usage.
*   **Flexible Training Options:**
    *   Multi-GPU training support.
    *   Aspect ratio bucketing for diverse image/video sizes.
    *   LoRA/LyCORIS and full UNet training options.
    *   EMA (Exponential Moving Average) for improved stability.
    *   S3-compatible storage support for training from cloud storage.
    *   ControlNet model training (LoRA or full) for SDXL, SD 1.x/2.x, and Flux.
    *   Mixture of Experts (MoE) training.
    *   Masked loss training.
    *   Prior regularisation support for LyCORIS models.
    *   Webhook support for progress updates.
    *   Hugging Face Hub integration for model uploads and model cards.

## Hardware Requirements

Detailed hardware requirements for each model are outlined below.  In general:

*   NVIDIA: 3080 and up is a safe bet.
*   AMD: LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.
*   Apple: M3 Max with 128GB memory.
*   See model-specific requirements below for more details.

### Model-Specific Hardware Requirements

*   **HiDream**:
    *   A100-80G (Full tune with DeepSpeed)
    *   A100-40G (LoRA, LoKr)
    *   3090 24G (LoRA, LoKr)

*   **Flux.1**:
    *   A100-80G (Full tune with DeepSpeed)
    *   A100-40G (LoRA, LoKr)
    *   3090 24G (LoRA, LoKr)
    *   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
    *   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

*   **Auraflow**:
    *   A100-80G (Full tune with DeepSpeed)
    *   A100-40G (LoRA, LoKr)
    *   3090 24G (LoRA, LoKr)
    *   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
    *   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

*   **SDXL, 1024px**:
    *   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
    *   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
    *   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
    *   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
    *   4080-12G (LoRA @ low-medium batch sizes)

*   **Stable Diffusion 2.x, 768px**:
    *   16GB or better

## Table of Contents

-   [Key Features](#key-features)
-   [Hardware Requirements](#hardware-requirements)
-   [Model-Specific Hardware Requirements](#model-specific-hardware-requirements)
-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Offers good default settings for most use cases.
*   **Versatility:** Handles various image quantities.
*   **Cutting-Edge Features:** Includes proven, effective features.

## Tutorial

Consult the [tutorial](/TUTORIAL.md) for comprehensive training guidance.  Use the [Quick Start](/documentation/QUICKSTART.md) guide for immediate implementation. See [DeepSpeed document](/documentation/DEEPSPEED.md) for memory constrained training. Find information on multi-node training configurations in [this guide](/documentation/DISTRIBUTED.md).

## Toolkit

Explore the [toolkit documentation](/toolkit/README.md) for related utility information.

## Setup

Follow the detailed setup instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for more insight: `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`) file.  For performance analysis: set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. Review [OPTIONS.md](/OPTIONS.md) for a comprehensive list of options.