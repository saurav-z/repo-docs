# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease

**SimpleTuner is your go-to solution for training diffusion models, offering simplicity and flexibility for both beginners and experienced users.** ([View on GitHub](https://github.com/bghira/SimpleTuner))

> ℹ️ No data is sent to any third parties except through opt-in features like `report_to`, `push_to_hub`, or webhooks which must be manually configured.

SimpleTuner prioritizes user-friendliness, making it easy to understand and customize your training process. Built as a shared academic exercise, contributions are always welcome. Join our community and discuss your projects on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group.

## Key Features

*   **Broad Model Support:** Train a wide range of models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict, and Qwen-Image.
*   **Versatile Training Options:** Supports LoRA, LyCORIS, full u-net, and ControlNet training for various models.
*   **Memory Optimization:** Includes techniques like aspect bucketing, pre-caching, DeepSpeed integration, and quantization to reduce VRAM usage.
*   **Advanced Techniques:** Features cutting-edge techniques like TREAD, EMA, Masked Loss training, and Mixture of Experts for enhanced performance and model stability.
*   **Hugging Face Integration:** Seamlessly integrate with the Hugging Face Hub for model uploading and dataset loading.
*   **S3 Compatible Training:** Train directly from S3-compatible storage providers, simplifying data access.
*   **Community Driven:** Benefit from a collaborative environment with active community support on Discord.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorials & Guides](#tutorials-guides)
-   [Features](#features)
-   [Hardware Requirements](#hardware-requirements)
-   [Scripts](#scripts)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Good default settings for ease of use with minimal configuration.
*   **Versatility:** Handles a broad spectrum of image and video quantities.
*   **Cutting-Edge:** Implements proven and effective features.

## Tutorials & Guides

*   Begin your journey with the main [Tutorial](/TUTORIAL.md).
*   For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.
*   Optimize for memory-constrained systems with [DeepSpeed documentation](/documentation/DEEPSPEED.md).
*   Configure multi-node distributed training using [this guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

*   [NVIDIA](#nvidia)
*   [AMD](#amd)
*   [Apple](#apple)
*   [HiDream](#hidream)
*   [Flux.1](#flux1-dev-schnell)
*   [Auraflow](#auraflow)
*   [SDXL, 1024px](#sdxl-1024px)
*   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)

### NVIDIA
Pretty much anything 3080 and up is a safe bet. YMMV.

### AMD
LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

Lacking `xformers`, it will use more memory than Nvidia equivalent hardware.

### Apple
LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.
  - You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.
  - Subscribing to Pytorch issues for MPS is probably a good idea, as random bugs will make training stop working.

### HiDream [dev, full]

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)

HiDream has not been tested on 16G cards, but with aggressive quantisation and pre-caching of embeds, you might make it work, though even 24G is pushing limits.

### Flux.1 [dev, schnell]

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

Flux prefers being trained with multiple large GPUs but a single 16G card should be able to do it with quantisation of the transformer and text encoders.

Kontext requires a bit beefier compute and memory allocation; a 4090 will go from ~3 to ~6 seconds per step when it is enabled.

### Auraflow

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

- A100-80G (EMA, large batches, LoRA @ insane batch sizes)
- A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
- A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
- 4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
- 4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

- 16G or better

## Scripts

(No changes from the original, as it is a directory)

## Toolkit

(No changes from the original, as it is a directory)
Refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enhance debugging by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

Analyze the training loop performance by setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`, which includes timestamp diagnostics.

For further option details, review [this documentation](/OPTIONS.md).