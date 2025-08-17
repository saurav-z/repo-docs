# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease 🚀

**SimpleTuner is an open-source project designed to simplify the process of training diffusion models, offering a streamlined and accessible experience for both beginners and experts.**

[View the original repository on GitHub](https://github.com/bghira/SimpleTuner)

> ℹ️ No data is sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

## Key Features

*   **User-Friendly Design:** Prioritizes simplicity with sensible defaults, minimizing the need for extensive configuration.
*   **Versatile Training:** Supports a wide range of image quantities and aspect ratios, from small datasets to large collections.
*   **Cutting-Edge Techniques:** Integrates proven features to enhance training efficiency and model quality.
*   **Multi-GPU Support:** Leverages multi-GPU capabilities for faster training.
*   **Memory Optimization:** Includes techniques like token-wise dropout (TREAD), caching, and DeepSpeed integration to reduce VRAM usage.
*   **Model Compatibility:** Supports a broad spectrum of models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, and Cosmos2 Predict.
*   **LoRA and Full Training:** Provides options for both LoRA (Low-Rank Adaptation) and full U-Net training.
*   **Quantization:** Offers support for quantizing models (NF4/INT8/FP8) to reduce VRAM requirements.
*   **S3 Storage Integration:** Enables training directly from S3-compatible storage providers.
*   **Hugging Face Hub Integration:** Facilitates seamless model upload and management via the Hugging Face Hub.
*   **ControlNet Training:** Offers training support for ControlNet models.
*   **Advanced Techniques:** Includes support for Mixture of Experts, Masked Loss Training, and Prior Regularisation.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity**: Easy to use with good default settings, reducing the need for extensive configuration.
*   **Versatility**: Designed to handle diverse image quantities and aspect ratios.
*   **Cutting-Edge Features**: Focused on features with proven efficacy.

## Tutorial

Begin by exploring this README for essential information before proceeding to the [tutorial](/TUTORIAL.md).

*   [Quick Start](/documentation/QUICKSTART.md)
*   [DeepSpeed document](/documentation/DEEPSPEED.md)
*   [Distributed Training](/documentation/DISTRIBUTED.md)

## Hardware Requirements

Hardware requirements vary depending on the model and training configuration. Consult the sections below for specific details.

### HiDream [dev, full]
*   A100-80G, A100-40G, 3090 24G

### Flux.1 [dev, schnell]
*   A100-80G, A100-40G, 3090 24G, 4060 Ti 16G, 4070 Ti 16G, 3080 16G, 4070 Super 12G, 3080 10G, 3060 12GB

### Auraflow
*   A100-80G, A100-40G, 3090 24G, 4060 Ti 16G, 4070 Ti 16G, 3080 16G, 4070 Super 12G, 3080 10G, 3060 12GB

### SDXL, 1024px
*   A100-80G, A6000-48G, A100-40G, 4090-24G, 4080-12G

### Stable Diffusion 2.x, 768px
*   16G or better

## Toolkit

The [toolkit documentation](/toolkit/README.md) provides additional information about SimpleTuner’s tools.

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs for more detailed insights.
*   `export SIMPLETUNER_LOG_LEVEL=DEBUG`
*   `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`

Consult [this documentation](/OPTIONS.md) for a comprehensive list of options.