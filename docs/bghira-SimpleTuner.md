# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease

**SimpleTuner** empowers you to train high-quality diffusion models, offering simplicity and versatility for both beginners and experts. Explore the power of AI image generation with this user-friendly toolkit!  [View the original repository on GitHub](https://github.com/bghira/SimpleTuner).

> ℹ️  No data is sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

## Key Features

*   **Simplified Training:**  Designed for ease of use with sensible default settings, reducing the need for extensive configuration.
*   **Versatile Dataset Handling:** Supports a wide range of image quantities and aspect ratios, from small datasets to massive collections.
*   **Cutting-Edge Techniques:** Integrates proven features to improve performance and results, including LoRA, LyCORIS, and ControlNet training.
*   **Multi-GPU Support:** Train faster with multi-GPU capabilities.
*   **Hardware-Efficient Training:** Supports various quantization techniques (NF4/INT8/FP8) and DeepSpeed integration to minimize VRAM usage, enabling training on 16GB and even 12GB GPUs.
*   **S3-Compatible Storage:** Train directly from cloud storage providers (e.g., Cloudflare R2, Wasabi S3) to avoid local storage limitations.
*   **Hugging Face Hub Integration:** Seamlessly upload and share your trained models with the Hugging Face community.
*   **Support for a Wide Range of Models:** Compatibility with popular models like HiDream, Flux.1, SD3, SDXL, PixArt Sigma, NVLabs Sana, LTX Video, and Cosmos2 Predict (image).
*   **Advanced Techniques:** Includes features like aspect bucketing, masked loss training, EMA (exponential moving average) and prior regularisation for improved training stability and outcomes.
*   **Customizable Workflows:** Support webhooks to track progress and debug with fine-grained debug logging.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
    -   [Flux](#flux1)
    -   [Wan 2.1 Video](#wan-video)
    -   [LTX Video](#ltx-video)
    -   [PixArt Sigma](#pixart-sigma)
    -   [NVLabs Sana](#nvlabs-sana)
    -   [Stable Diffusion 2.0/2.1](#stable-diffusion-20--21)
    -   [Stable Diffusion 3.0](#stable-diffusion-3)
    -   [Kwai Kolors](#kwai-kolors)
    -   [Lumina2](#lumina2)
-   [Hardware Requirements](#hardware-requirements)
    -   [Flux](#flux1-dev-schnell)
    -   [SDXL](#sdxl-1024px)
    -   [Stable Diffusion (Legacy)](#stable-diffusion-2x-768px)
-   [Scripts](#scripts)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity:**  Prioritizes user-friendliness with optimized default settings.
-   **Versatility:** Accommodates diverse datasets and aspect ratios.
-   **Cutting-Edge:** Focuses on proven, effective features.

## Tutorial

Before getting started, please review the [tutorial](/TUTORIAL.md) for important information. For a quick start, refer to the [Quick Start](/documentation/QUICKSTART.md) guide.  For memory-constrained systems, explore the [DeepSpeed document](/documentation/DEEPSPEED.md). For multi-node distributed training configuration, consult the [guide](/documentation/DISTRIBUTED.md).

## Features

**(See Key Features section for a summarized view)**

### HiDream

-   Custom ControlNet implementation (full-rank, LoRA, Lycoris).
-   Memory-efficient training for NVIDIA GPUs (AMD support planned).
-   Optional MoEGate loss augmentation.
-   Quantization options for memory savings.

### Flux.1

-   Double the training speed with `--fuse_qkv_projections`.
-   ControlNet training (full-rank, LoRA, Lycoris).
-   Instruct fine-tuning for Kontext.
-   Classifier-free guidance training options.
-   T5 attention masked training.
-   Quantization for memory savings.

### Wan Video

-   Text to Video training support.
-   LyCORIS, PEFT, and full tuning supported.

### LTX Video

-   LyCORIS, PEFT, and full tuning supported.

### PixArt Sigma

-   Text encoder training is not supported.
-   LyCORIS and full tuning work as expected.
-   ControlNet training is supported for full and PEFT LoRA training.
-   Two-stage PixArt training support.

### NVLabs Sana

-   LyCORIS and full tuning work as expected.
-   Not supporting: Text encoder training, PEFT Standard LoRA, ControlNet training.

### Stable Diffusion 3

-   LoRA and full finetuning supported.
-   ControlNet training support (full-rank, PEFT LoRA, Lycoris).

### Kwai Kolors

-   An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, **doubling** the hidden dimension size.

### Lumina2

-   LoRA, Lycoris, and full finetuning supported.

### Cosmos2 Predict (Image)

-   Currently, only the text-to-image variant is supported.
-   Lycoris or full-rank tuning are supported, but PEFT LoRAs are currently not.

### Qwen-Image

-   Lycoris, LoRA, and full-rank training are all supported, with full-rank training requiring H200 or better with DeepSpeed

### Legacy Stable Diffusion models

-   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

## Hardware Requirements

### NVIDIA

*(General guidelines)* 3080 and up is a safe bet.  YMMV.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

Lacking `xformers`, it will use more memory than Nvidia equivalent hardware.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.
  - You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.

### HiDream [dev, full]

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)

### Flux.1 [dev, schnell]

- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

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

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md) for details.

## Setup

Follow the [installation documentation](/INSTALL.md) for setup instructions.

## Troubleshooting

Enable debug logs for detailed insights: `export SIMPLETUNER_LOG_LEVEL=DEBUG`. Analyze training loop performance using: `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.  Consult [this documentation](/OPTIONS.md) for a comprehensive list of options.