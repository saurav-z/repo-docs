# SimpleTuner: Fine-tune Diffusion Models with Ease 💹

**SimpleTuner empowers you to easily fine-tune a wide range of diffusion models, from SDXL to HiDream, with a focus on simplicity and cutting-edge features.** ([Original Repository](https://github.com/bghira/SimpleTuner))

> ℹ️ No data is sent to any third parties except through opt-in features and webhooks configured by the user.

SimpleTuner is designed for ease of use and aims to provide excellent default settings for most training scenarios. Contributions are welcome! Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for support and collaboration.

## Key Features

*   **Broad Model Support:** Fine-tune popular models including SDXL, Stable Diffusion 3, HiDream, Flux.1, and many more (see below).
*   **Memory Optimization:** Features like aspect bucketing, caching, and DeepSpeed integration reduce VRAM usage for training on a wide range of GPUs (even down to 16GB).
*   **Versatile Training Options:** Supports LoRA/LyCORIS, full u-net training, ControlNet, Mixture of Experts, and masked loss training.
*   **Cutting-Edge Techniques:** Includes advanced techniques like TREAD for faster training and prior regularization for improved performance.
*   **Hugging Face Hub Integration:** Seamlessly integrates with the Hugging Face Hub for dataset loading and model uploads.
*   **S3 Storage Support:** Train directly from S3-compatible storage providers, removing the need for local storage.
*   **Webhook Support:** Receive real-time updates on your training progress via webhooks (e.g., Discord).

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

*   **Simplicity:** Designed with easy-to-use, good default settings.
*   **Versatility:** Supports various image/video sizes and aspect ratios, from small datasets to large collections.
*   **Cutting-Edge Features:** Only incorporates proven features.

## Tutorial

Refer to the [tutorial](/TUTORIAL.md) for detailed guidance.

*   **Quick Start:** Get started quickly with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Use [DeepSpeed](/documentation/DEEPSPEED.md) for memory-constrained systems.
*   **Distributed Training:** Optimize for multi-node training with [this guide](/documentation/DISTRIBUTED.md).

## Features (Detailed)

*(See the original README for detailed feature descriptions of each model.)*

### HiDream

*   Custom ControlNet implementation
*   Memory-efficient training
*   MoEGate loss augmentation (optional)
*   Lycoris or full tuning via DeepSpeed ZeRO
*   Quantization support

### Flux.1

*   Double the training speed with `--fuse_qkv_projections`
*   ControlNet training
*   Instruct fine-tuning for the Kontext editing model
*   Classifier-free guidance training
*   T5 attention masked training (optional)
*   LoRA or full tuning via DeepSpeed ZeRO
*   Quantization support

### Wan Video

*   Text to Video training support
*   LyCORIS, PEFT, and full tuning support

### LTX Video

*   LyCORIS, PEFT, and full tuning support

### PixArt Sigma

*   Text encoder training is not supported
*   LyCORIS and full tuning
*   ControlNet training supported
*   Two-stage PixArt support

### NVLabs Sana

*   LyCORIS and full tuning supported

### Stable Diffusion 3

*   LoRA and full finetuning support
*   ControlNet training support

### Kwai Kolors

*   SDXL-based with ChatGLM (General Language Model) 6B text encoder

### Lumina2

*   LoRA, Lycoris, and full finetuning supported

### Cosmos2 Predict (Image)

*   Text-to-image variant supported
*   Lycoris or full-rank tuning supported

### Qwen-Image

*   Lycoris, LoRA, and full-rank training supported
*   ControlNet training not supported

### Legacy Stable Diffusion models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

*(Refer to the original README for specific hardware recommendations for each model.)*

*   **NVIDIA:** 3080 and up are recommended.
*   **AMD:** LoRA and full-rank tuning verified on 7900 XTX and MI300X.
*   **Apple:** Tested on M3 Max with 128GB memory.
*   **Specific model requirements:** (See below)

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

For more information, see [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG`. For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. Consult [this documentation](/OPTIONS.md) for a comprehensive list of options.