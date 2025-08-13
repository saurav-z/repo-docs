# SimpleTuner: Simplified and Versatile Diffusion Model Training

> **SimpleTuner empowers you to easily train cutting-edge diffusion models, focusing on simplicity and versatility.**

This project provides a straightforward, academic-focused codebase for training diffusion models, prioritizing ease of understanding and use. Contributions are welcome! No data is sent to third parties unless explicitly enabled through opt-in flags or manually configured webhooks.

🔗 [View the original repository on GitHub](https://github.com/bghira/SimpleTuner)

## Key Features

*   🚀 **Simplified Design:** Easy to understand code for accessible model training.
*   🖼️ **Versatile:** Supports a wide range of image quantities and aspect ratios, from small datasets to large collections.
*   💡 **Cutting-Edge:** Includes the latest, proven training techniques.
*   💻 **Multi-GPU Support:** Train faster with multi-GPU setups.
*   ⚡ **Performance Enhancements:** Features image/video caching and token-wise dropout (TREAD) for accelerated training.
*   🎨 **Wide Model Compatibility:** Supports training for:
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
    *   Legacy Stable Diffusion models
*   💾 **VRAM Optimization:** Includes options for quantisation, DeepSpeed integration, and EMA to reduce VRAM consumption.
*   ☁️ **Cloud Storage Integration:** Train directly from S3-compatible storage providers (e.g., Cloudflare R2, Wasabi S3).
*   🌐 **Hugging Face Hub Integration:** Seamless model upload and model card generation.
*   🔔 **Webhook Support:** Get training progress updates, validations, and error notifications via webhooks (e.g., Discord).

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
-   [Hardware Requirements](#hardware-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizing easy-to-use defaults.
*   **Versatility:** Accommodating datasets of varying sizes and aspect ratios.
*   **Cutting-Edge:** Incorporating only proven features.

## Tutorial

Please explore this README and the [tutorial](/TUTORIAL.md) for essential information.

*   **Quick Start:** Begin quickly with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Optimize for memory-constrained systems with the [DeepSpeed document](/documentation/DEEPSPEED.md).
*   **Multi-node Training:** Adjust configurations for multi-node training with [this guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

### NVIDIA

*   Generally safe bet with anything 3080 or up.

### AMD

*   LoRA and full-rank tuning have been verified working on a 7900 XTX 24GB and MI300X.
*   More memory usage may be necessary due to the absence of `xformers`.

### Apple

*   LoRA and full-rank tuning have been tested to work on an M3 Max with 128GB of memory
*   24GB+ recommended for machine learning due to the lack of memory-efficient attention.

### HiDream

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)

### Flux.1

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

Refer to [the toolkit documentation](/toolkit/README.md) for more information.

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG` (in your environment config).

Analyze training loop performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a comprehensive list of options, consult [this documentation](/OPTIONS.md).