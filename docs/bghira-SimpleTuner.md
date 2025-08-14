# SimpleTuner: Simplify Your AI Model Training 🚀

SimpleTuner is designed for ease of use, making AI model training accessible for everyone. **Train cutting-edge diffusion models with simplicity and efficiency using SimpleTuner.**  [View the original repository on GitHub](https://github.com/bghira/SimpleTuner).

*   **Privacy Focused:**  No data is sent to third parties unless you explicitly opt-in via flags or manually configured webhooks.

## Key Features

*   **Versatile Model Support:** Train a wide range of diffusion models, including:
    *   HiDream
    *   Flux.1
    *   Wan Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3.0
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   Qwen-Image
    *   Legacy Stable Diffusion models (SD 1.x/2.x)
*   **Optimized for Performance:**
    *   Multi-GPU training support.
    *   Advanced techniques like TREAD (token-wise dropout) for faster training.
    *   Caching of image/video features for reduced memory usage and faster training.
    *   Aspect bucketing to handle diverse image and video sizes.
*   **Memory Efficiency:**
    *   LoRA/LyCORIS training for several models, enabling training on GPUs with limited VRAM (e.g., 16GB).
    *   DeepSpeed integration for offloading optimizer state.
    *   Quantization (NF4/INT8/FP8) to reduce VRAM consumption.
*   **Advanced Training Techniques:**
    *   Optional EMA (Exponential Moving Average) weight networks.
    *   Training directly from S3-compatible storage.
    *   Full or LoRA-based ControlNet model training.
    *   Mixture of Experts training support.
    *   Masked loss training for improved results.
    *   Prior regularization support.
*   **Integration & Tools:**
    *   Webhook support for progress updates (e.g., Discord integration).
    *   Hugging Face Hub integration for model sharing.
    *   Comprehensive toolkit for model analysis and more.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
-   [Hardware Requirements](#hardware-requirements)
-   [Scripts](#scripts)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity:** User-friendly defaults for ease of use.
-   **Versatility:** Handles a wide range of datasets.
-   **Cutting-Edge:** Incorporates proven features for optimal performance.

## Tutorial

Before getting started, explore the [tutorial](/TUTORIAL.md) for important information.

*   **Quick Start:**  Get up and running quickly with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Configure DeepSpeed for memory-constrained systems with the [DeepSpeed document](/documentation/DEEPSPEED.md).
*   **Distributed Training:** Optimize multi-node training using the [distributed training guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

Comprehensive details in the original README, covering:
*   NVIDIA
*   AMD
*   Apple
*   HiDream
*   Flux.1
*   Auraflow
*   SDXL, 1024px
*   Stable Diffusion 2.x, 768px

## Scripts

Refer to the [README.md](/toolkit/README.md) for more about the associated toolkit.

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logging with `export SIMPLETUNER_LOG_LEVEL=DEBUG` to analyze issues. For training loop performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. See [OPTIONS.md](/OPTIONS.md) for a full list of options.