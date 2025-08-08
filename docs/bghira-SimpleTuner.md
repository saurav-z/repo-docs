# SimpleTuner: Train Cutting-Edge AI Models with Ease

> **SimpleTuner empowers you to fine-tune a variety of AI models with simplicity and efficiency.**

[Original Repo](https://github.com/bghira/SimpleTuner)

SimpleTuner is designed for ease of use, making it a great choice for both beginners and experienced practitioners. This project is an open-source academic exercise, and contributions are welcome.  Please note that no data is sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group to ask questions and share your work.

## Key Features

*   **Simplified Training:** Get started quickly with good default settings.
*   **Versatile:** Supports a wide range of image quantities and aspect ratios.
*   **Cutting-Edge:** Incorporates proven techniques and features.
*   **Multi-GPU Training:** Leverage multiple GPUs for faster training.
*   **Advanced Techniques:**  Includes support for new token-wise dropout (TREAD), aspect bucketing, and EMA.
*   **Memory Optimization:** Image, video, and caption caching to reduce memory consumption. DeepSpeed integration for memory-constrained systems.  Quantization options to reduce VRAM usage.
*   **Model Support:** Fine-tune a wide variety of models including:
    *   HiDream
    *   Flux.1
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   Qwen-Image
    *   Legacy Stable Diffusion Models (SD 1.5 & 2.x)
*   **Storage Flexibility:** Train directly from S3-compatible storage providers.
*   **Hugging Face Integration:** Seamlessly upload and share your models.
*   **Webhook Support:** Integrate with external services like Discord for training progress updates.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Model-Specific Features](#model-specific-features)
    *   [HiDream](#hidream)
    *   [Flux.1](#flux1)
    *   [Wan Video](#wan-video)
    *   [LTX Video](#ltx-video)
    *   [PixArt Sigma](#pixart-sigma)
    *   [NVLabs Sana](#nvlabs-sana)
    *   [Stable Diffusion 3](#stable-diffusion-3)
    *   [Kwai Kolors](#kwai-kolors)
    *   [Lumina2](#lumina2)
    *   [Cosmos2 Predict (Image)](#cosmos2-predict-image)
    *   [Qwen-Image](#qwen-image)
    *   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** SimpleTuner prioritizes ease of use with sensible defaults.
*   **Versatility:** Handles various image quantities, sizes, and aspect ratios.
*   **Cutting-Edge Features:** Focus on proven features for effective training.

## Tutorial

Review this README and then explore the [tutorial](/TUTORIAL.md) for comprehensive information.

For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory optimization with DeepSpeed, see the [DeepSpeed document](/documentation/DEEPSPEED.md).

For multi-node distributed training, consult the [guide](/documentation/DISTRIBUTED.md).

## Model-Specific Features

Detailed features and quickstart guides are available for each supported model.

### HiDream

*   Custom ControlNet implementation (full-rank, LoRA, or Lycoris)
*   Memory-efficient training for NVIDIA GPUs (AMD support planned)
*   Optional MoEGate loss augmentation
*   Quantization options for memory savings

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Double the training speed with `--fuse_qkv_projections` (Hopper systems)
*   ControlNet training (full-rank, LoRA, or Lycoris)
*   Instruct fine-tuning for Kontext
*   Classifier-free guidance training
*   T5 attention masked training (optional)
*   Quantization options for memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text-to-Video training support
*   LyCORIS, PEFT, and full tuning
*   ControlNet training not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide.

### LTX Video

*   LyCORIS, PEFT, and full tuning
*   ControlNet training not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide.

### PixArt Sigma

*   Text encoder training not supported
*   LyCORIS and full tuning
*   ControlNet training (full and PEFT LoRA)
*   Two-stage PixArt training support

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.

### NVLabs Sana

*   LyCORIS and full tuning
*   Text encoder training not supported
*   ControlNet training not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.

### Stable Diffusion 3

*   LoRA and full finetuning supported
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris
*   Parameters optimized for best results

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

*   SDXL-based model with ChatGLM (General Language Model) 6B
*   Doubles hidden dimension size and increases local detail in prompt embeds

### Lumina2

*   2B parameter flow-matching model (Flux VAE)
*   LoRA, Lycoris, and full finetuning
*   ControlNet training not yet supported

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available.

### Cosmos2 Predict (Image)

*   2B / 14B parameter model for text-to-image.
*   Lycoris or full-rank tuning are supported
*   ControlNet training not yet supported.

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available.

### Qwen-Image

*   20B MMDiT for text-to-image.
*   Lycoris, LoRA, and full-rank training
*   ControlNet training not yet supported.

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available.

### Legacy Stable Diffusion models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x supported.

---

## Hardware Requirements

Hardware requirements vary depending on the model and configuration.  See the specific model sections above for more details.

### NVIDIA

Generally, any NVIDIA GPU from the 3080 and up will be a safe bet.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

### Apple

LoRA and full-rank tuning tested to work on an M3 Max with 128GB of memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.

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

---

## Toolkit

The SimpleTuner toolkit documentation can be found [here](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment file (e.g., `config/config.env`).

For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a comprehensive list of options, see [this documentation](/OPTIONS.md).