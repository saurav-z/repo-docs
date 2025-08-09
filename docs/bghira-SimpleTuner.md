# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease 🚀

SimpleTuner is a user-friendly toolkit designed to simplify training various diffusion models, offering a streamlined experience for both beginners and experienced users. [Visit the GitHub repository](https://github.com/bghira/SimpleTuner) to get started.

**Key Features:**

*   **Wide Model Support:** Train a diverse range of models, including HiDream, Flux.1, SDXL, Stable Diffusion 3, PixArt Sigma, and many more.
*   **Simplified Training:** Designed with simplicity in mind, offering good default settings to minimize the need for extensive configuration.
*   **Memory Optimization:** Features like caching, aspect bucketing, and DeepSpeed integration for efficient training on limited hardware.
*   **Flexible Training Options:** Supports LoRA, LyCORIS, full finetuning, and ControlNet training for various models.
*   **Advanced Techniques:** Implements cutting-edge techniques such as TREAD, Mixture of Experts, and masked loss training for superior results.
*   **Hardware Compatibility:** Supports a variety of GPUs from NVIDIA, AMD, and Apple Silicon (M-series).
*   **Community and Support:** Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for questions and support.

**Important Notes:**

*   No data is sent to third parties by default. Opt-in features like `report_to`, `push_to_hub`, and webhooks require manual configuration.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
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

*   **Simplicity:** Good default settings minimize the need for configuration.
*   **Versatility:** Handles a wide range of image quantities.
*   **Cutting-Edge Features:** Incorporates proven effective features.

## Tutorial

Explore the [tutorial](/TUTORIAL.md) for comprehensive information.

For quick starts, refer to the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory-constrained systems, see the [DeepSpeed document](/documentation/DEEPSPEED.md).

For multi-node distributed training, see [this guide](/documentation/DISTRIBUTED.md).

## Features

*   Multi-GPU training
*   New token-wise dropout techniques like [TREAD](/documentation/TREAD.md)
*   Image, video, and caption caching
*   Aspect bucketing
*   Refiner LoRA or full u-net training for SDXL
*   Training on 16G or 24G GPUs
*   DeepSpeed integration
*   Quantised NF4/INT8/FP8 LoRA training
*   Optional EMA
*   Training directly from S3-compatible storage
*   ControlNet model training for SDXL, SD 1.x/2.x, and Flux
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md)
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss)
*   [Prior regularisation](/documentation/DATALOADER.md#is_regularisation_data)
*   Webhook support
*   Integration with the [Hugging Face Hub](https://huggingface.co)
*   Use the [datasets library](/documentation/data_presets/preset_subjects200k.md) to load compatible datasets directly from the hub

### HiDream

*   Custom ControlNet implementation
*   Memory-efficient training for NVIDIA GPUs (AMD support is planned)
*   Lycoris or full tuning via DeepSpeed ZeRO
*   Quantisation of base model using `--base_model_precision`
*   Quantise Llama LLM using `--text_encoder_4_precision`

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Fuse QKV projections for training speed
*   ControlNet training
*   Instruct fine-tuning
*   Classifier-free guidance training
*   (optional) T5 attention masked training
*   LoRA or full tuning via DeepSpeed ZeRO
*   Quantisation of base model using `--base_model_precision`

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text to Video training support
*   LyCORIS, PEFT, and full tuning all work as expected

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide.

### LTX Video

*   LyCORIS, PEFT, and full tuning all work as expected

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide.

### PixArt Sigma

*   Text encoder training is not supported
*   LyCORIS and full tuning work as expected
*   ControlNet training is supported
*   [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.

### NVLabs Sana

*   LyCORIS and full tuning work as expected.

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.

### Stable Diffusion 3

*   LoRA and full finetuning support
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).

### Kwai Kolors

*   SDXL-based model
*   ChatGLM (General Language Model) 6B as its text encoder
*   No ControlNet training support

### Lumina2

*   2B parameter flow-matching model
*   LoRA, Lycoris, and full finetuning are supported
*   No ControlNet training support

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available.

### Cosmos2 Predict (Image)

*   2B / 14B parameter model
*   Lycoris or full-rank tuning are supported
*   No ControlNet training support

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available.

### Qwen-Image

*   20B MMDiT
*   Lycoris, LoRA, and full-rank training are all supported
*   No ControlNet training support

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available.

### Legacy Stable Diffusion models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable

---

## Hardware Requirements

### NVIDIA

Generally, 3080 and up.

### AMD

LoRA and full-rank tuning verified working on a 7900 XTX 24GB and MI300X.

### Apple

Tested to work on an M3 Max with 128G memory.

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

*   A100-80G
*   A6000-48G
*   A100-40G
*   4090-24G
*   4080-12G

### Stable Diffusion 2.x, 768px

*   16G or better

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md).

## Setup

See the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment.

For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a comprehensive list of options, consult [this documentation](/OPTIONS.md).