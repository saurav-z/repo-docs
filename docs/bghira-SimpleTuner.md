# SimpleTuner: Unlock Your AI Art Potential with Easy Diffusion Model Training

**SimpleTuner** is an open-source project designed to simplify the process of training diffusion models, making it accessible to everyone. ([View on GitHub](https://github.com/bghira/SimpleTuner))

Key Features:

*   **Simplified Training:** Focuses on ease of use with sensible defaults, minimizing the need for complex configurations.
*   **Versatile:** Supports training on various image and video datasets, from small to massive.
*   **Cutting-Edge Technology:** Integrates proven features and techniques to optimize training performance.
*   **Multi-Model Support:**  Comprehensive support for a wide range of diffusion models including HiDream, Flux.1, SDXL, Stable Diffusion 3, PixArt Sigma, NVLabs Sana, and more.
*   **Memory Optimization:** Offers techniques like DeepSpeed integration, quantisation, and caching to reduce VRAM usage.
*   **Flexible Training Options:** Supports LoRA, LyCORIS, full u-net training, ControlNet, and Mixture of Experts.
*   **Cloud and Hub Integration:** Trains directly from S3-compatible storage and integrates with the Hugging Face Hub for seamless model uploading.
*   **Community Driven:** Open to contributions from the community, [join us on Discord!](https://discord.gg/CVzhX7ZA)

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorials & Quick Start](#tutorial)
*   [Features](#features)
    *   [HiDream](#hidream)
    *   [Flux.1](#flux-1)
    *   [Wan Video](#wan-video)
    *   [LTX Video](#ltx-video)
    *   [PixArt Sigma](#pixart-sigma)
    *   [NVLabs Sana](#nvlabs-sana)
    *   [Stable Diffusion 3](#stable-diffusion-3)
    *   [Kwai Kolors](#kwai-kolors)
    *   [Lumina2](#lumina2)
    *   [Cosmos2 Predict](#cosmos2-predict)
    *   [Qwen-Image](#qwen-image)
    *   [Legacy Stable Diffusion Models](#legacy-stable-diffusion-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizes ease of use with sensible default settings to minimize manual configuration.
*   **Versatility:** Supports a wide range of image quantities, from small datasets to extensive collections.
*   **Cutting-Edge:** Incorporates only proven, effective features, avoiding untested options.

## Tutorials & Quick Start

Start your AI art journey with the [Quick Start guide](/documentation/QUICKSTART.md) or explore the comprehensive [tutorial](/TUTORIAL.md) to understand the nuances of SimpleTuner. For memory-constrained systems, refer to the [DeepSpeed document](/documentation/DEEPSPEED.md). For multi-node distributed training, consult the [distributed training guide](/documentation/DISTRIBUTED.md).

## Features

*   Multi-GPU Training
*   New token-wise dropout techniques like [TREAD](/documentation/TREAD.md) for accelerating training
*   Image, video, and caption caching for faster, more memory-efficient training
*   Aspect bucketing support for training various image/video sizes
*   Refiner LoRA or full u-net training for SDXL
*   Train models on GPUs with 16GB or less VRAM
*   DeepSpeed integration for training large models on limited VRAM
*   Quantised NF4/INT8/FP8 LoRA training for reduced VRAM consumption
*   Optional EMA (Exponential Moving Average) for improved training stability
*   Train directly from S3-compatible storage
*   Full or LoRA based [ControlNet model training](/documentation/CONTROLNET.md)
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md)
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss)
*   Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data)
*   Webhook support for tracking training progress and errors
*   Integration with the [Hugging Face Hub](https://huggingface.co)

### HiDream

*   Custom ControlNet implementation.
*   Memory-efficient training.
*   MoEGate loss augmentation.
*   Lycoris or full tuning via DeepSpeed ZeRO.
*   Quantisation for memory savings.

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Double training speed with `--fuse_qkv_projections`.
*   ControlNet training.
*   Instruct fine-tuning.
*   Classifier-free guidance training.
*   T5 attention masked training.
*   LoRA or full tuning via DeepSpeed ZeRO.
*   Quantisation for memory savings.

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text to Video training support
*   LyCORIS, PEFT, and full tuning
*   See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

*   LyCORIS, PEFT, and full tuning
*   See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide.

### PixArt Sigma

*   LyCORIS and full tuning
*   ControlNet training
*   [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training
*   See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.

### NVLabs Sana

*   LyCORIS and full tuning
*   See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.

### Stable Diffusion 3

*   LoRA, full finetuning and ControlNet training
*   See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).

### Kwai Kolors

*   SDXL-based model with ChatGLM text encoder.

### Lumina2

*   LoRA, Lycoris, and full finetuning
*   See a [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md)

### Cosmos2 Predict

*   Lycoris or full-rank tuning
*   See a [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md)

### Qwen-Image

*   Lycoris, LoRA, and full-rank training supported
*   See a [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md)

### Legacy Stable Diffusion Models

*   Support for RunwayML's SD 1.5 and StabilityAI's SD 2.x
---

## Hardware Requirements

Refer to the detailed hardware requirements for optimal performance:

*   [NVIDIA](#nvidia)
*   [AMD](#amd)
*   [Apple](#apple)
*   [HiDream](#hidream)
*   [Flux.1](#flux-1)
*   [SDXL, 1024px](#sdxl-1024px)
*   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed installation and setup instructions can be found in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/OPTIONS.md).