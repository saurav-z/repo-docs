# SimpleTuner: Unleash Your Creativity with Simplified Diffusion Model Training 🚀

**SimpleTuner** is a user-friendly toolkit designed to simplify the training of diffusion models, making it accessible for both beginners and experienced users. [Check out the original repo](https://github.com/bghira/SimpleTuner)!

> ℹ️ No data is sent to any third parties except through opt-in flags `report_to`, `push_to_hub`, or webhooks which must be manually configured.

## Key Features

*   **Simplified Training:** Easy-to-understand codebase with sensible default settings for quick experimentation and reduced tinkering.
*   **Versatile Support:** Handles a wide range of image and video quantities, from small datasets to vast collections.
*   **Cutting-Edge Techniques:** Integrates proven features for enhanced performance and stability.
*   **Multi-GPU Training:** Accelerate your training with multi-GPU support.
*   **Diverse Model Compatibility:**
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
    *   Legacy Stable Diffusion models (SD 1.5, 2.x)
*   **Advanced Techniques:** Supports techniques like TREAD, aspect bucketing, EMA, quantized training, ControlNet, Mixture of Experts, and masked loss training.
*   **Flexible Data Loading:** Train directly from S3-compatible storage providers (e.g., Cloudflare R2, Wasabi S3) and Hugging Face Hub datasets.
*   **DeepSpeed Integration:** Leverage DeepSpeed for memory-efficient training on resource-constrained systems.
*   **Webhook Support:** Integrate with webhooks for progress updates, validations, and error notifications.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Hardware Requirements](#hardware-requirements)
*   [Scripts](#scripts)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Focus on ease of use with sensible default settings.
*   **Versatility:** Support for diverse datasets and image quantities.
*   **Performance:** Integrate only features that have proven efficacy.

## Tutorial

Start your journey with the [Quick Start](/documentation/QUICKSTART.md) guide.

For more in-depth information, explore the full [Tutorial](/TUTORIAL.md).

Optimize for memory-constrained systems with [DeepSpeed document](/documentation/DEEPSPEED.md).

Learn about multi-node distributed training using [this guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

Refer to the following for hardware requirements:

*   [NVIDIA](#nvidia)
*   [AMD](#amd)
*   [Apple](#apple)
*   [HiDream](#hidream)
*   [Flux.1](#flux1-dev-schnell)
*   [Auraflow](#auraflow)
*   [SDXL, 1024px](#sdxl-1024px)
*   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)

## Toolkit

For information about the SimpleTuner toolkit, see [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs via `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`) for detailed insights.

For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a comprehensive list of options, consult [OPTIONS.md](/OPTIONS.md).