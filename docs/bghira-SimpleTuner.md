# SimpleTuner: Train Diffusion Models with Ease 💹

**SimpleTuner is a streamlined toolkit designed to simplify and accelerate the training of various diffusion models, empowering you to create high-quality AI-generated content.** ([Original Repo](https://github.com/bghira/SimpleTuner))

> ℹ️ No data is sent to third parties except through opt-in flags (`report_to`, `push_to_hub`) or manually configured webhooks.

**Key Features:**

*   **Simplified Training:** Focuses on ease of use with sensible defaults, reducing the need for extensive configuration.
*   **Versatile:** Supports a wide range of image/video quantities, from small datasets to massive collections.
*   **Cutting-Edge Technology:** Integrates proven techniques for optimal performance, avoiding unnecessary complexity.
*   **Multi-GPU Training:**  Leverages multiple GPUs for faster training.
*   **Advanced Techniques:**
    *   New token-wise dropout techniques (TREAD) for optimized performance.
    *   Caching for faster training and reduced memory consumption.
    *   Aspect ratio handling for diverse image/video sizes.
    *   LoRA/LyCORIS and full u-net training options.
    *   DeepSpeed integration for memory-constrained systems.
    *   Quantized training for reduced VRAM usage (NF4/INT8/FP8).
    *   EMA (Exponential Moving Average) for stability and reduced overfitting.
    *   Training from S3-compatible storage.
    *   ControlNet training for various models.
    *   Mixture of Experts (MoE) support.
    *   Masked loss training.
    *   Prior regularization support.
    *   Webhook support for progress updates.
    *   Hugging Face Hub integration for easy model sharing.

**Supported Models:**

*   HiDream
*   Flux.1
*   Wan Video
*   LTX Video
*   PixArt Sigma
*   NVLabs Sana
*   Stable Diffusion 3
*   Kwai Kolors
*   Lumina2
*   Cosmos2 Predict (Image)
*   Qwen-Image
*   Legacy Stable Diffusion (SD 1.5 & 2.x)

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorials](#tutorial)
*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Easy-to-use with good default settings.
*   **Versatility:** Handles diverse image/video quantities.
*   **Cutting-Edge:** Integrates proven features.

## Tutorials

*   Explore the complete [Tutorial](/TUTORIAL.md) to begin, containing vital information.
*   Start quickly with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   Optimize for memory-constrained systems using [DeepSpeed](/documentation/DEEPSPEED.md).
*   Set up multi-node distributed training via [this guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

*   **NVIDIA:**  Generally, 3080 and up.
*   **AMD:** LoRA and full-rank tuning are verified on a 7900 XTX 24GB and MI300X. (More memory usage than Nvidia)
*   **Apple:** Requires 24GB+ M-series hardware due to memory usage, tested on M3 Max (128GB).
*   **Specific Models:** Hardware recommendations are provided for each [supported model](#supported-models).

## Toolkit

Refer to the [toolkit documentation](/toolkit/README.md) for more information.

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment file (`config/config.env`). For performance analysis use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.  A comprehensive list of options is available in [this documentation](/OPTIONS.md).