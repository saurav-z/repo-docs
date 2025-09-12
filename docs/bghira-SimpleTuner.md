# SimpleTuner: Your Simple Solution for Training Diffusion Models 🚀

SimpleTuner simplifies the process of training diffusion models, making it easy to understand and use for everyone. Find the original repo [here](https://github.com/bghira/SimpleTuner).

**Key Features:**

*   **Ease of Use:** Designed for simplicity with good default settings.
*   **Versatile:** Supports various image quantities and aspect ratios, including wide and portrait formats.
*   **Cutting-Edge:** Integrates proven features like LoRA, LyCORIS, and ControlNet training.
*   **Hardware Optimization:** Provides options for multi-GPU training, DeepSpeed integration, and quantisation to reduce VRAM consumption.
*   **Integration:** Compatible with Hugging Face Hub, S3-compatible storage, and webhooks for progress updates.
*   **Model Support:** Extensive support for a wide range of models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict (Image), and Qwen-Image.

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

*   **Simplicity:** Easy to use with sensible defaults.
*   **Versatility:** Supports diverse datasets and aspect ratios.
*   **Innovation:** Integrates features with proven efficacy.

## Tutorial

Start with the [Quick Start](/documentation/QUICKSTART.md) guide. For detailed information, see the [main tutorial](/TUTORIAL.md).

## Features

*   Multi-GPU training
*   New token-wise dropout techniques like [TREAD](/documentation/TREAD.md)
*   Image, video, and caption caching for faster training
*   Aspect bucketing for diverse image/video sizes
*   Refiner LoRA or full u-net training for SDXL
*   Memory-efficient training on GPUs, including 16G options
*   DeepSpeed integration (e.g., training SDXL on 12G VRAM)
*   Quantised NF4/INT8/FP8 LoRA training
*   Optional EMA (Exponential moving average) weight network
*   Train directly from S3-compatible storage
*   ControlNet model training for SDXL, SD 1.x/2.x, and Flux
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight models
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss)
*   Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data)
*   Webhook support for real-time updates
*   Integration with the [Hugging Face Hub](https://huggingface.co)

### HiDream

*   Custom ControlNet implementation.
*   Memory-efficient training for NVIDIA GPUs.
*   Lycoris or full tuning via DeepSpeed ZeRO.
*   Quantisation for memory savings.

### Flux.1

*   Training speed doubled with `--fuse_qkv_projections`.
*   ControlNet, Instruct fine-tuning, and classifier-free guidance.
*   LoRA or full tuning via DeepSpeed ZeRO.
*   Quantisation for reduced VRAM.

### Wan Video

*   Text to Video training.
*   LyCORIS, PEFT, and full tuning support.

### LTX Video

*   LyCORIS, PEFT, and full tuning support.

### PixArt Sigma

*   Text encoder training is not supported
*   LyCORIS and full tuning support
*   ControlNet training support
*   Two-stage PixArt training

### NVLabs Sana

*   LyCORIS and full tuning support.

### Stable Diffusion 3

*   LoRA and full finetuning support.
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris

### Kwai Kolors

*   SDXL-based model with ChatGLM 6B.
*   LyCORIS, LoRA, and full finetuning are supported.

### Lumina2

*   LoRA, Lycoris, and full finetuning support.

### Cosmos2 Predict (Image)

*   Text-to-image variant supported.
*   Lycoris or full-rank tuning supported.

### Qwen-Image

*   Massive 20B MMDiT model for text-to-image.
*   Lycoris, LoRA, and full-rank training support.

### Legacy Stable Diffusion models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

## Hardware Requirements

*   **NVIDIA:** Generally, 3080 and up.
*   **AMD:** Verified on 7900 XTX 24GB and MI300X (may require more memory).
*   **Apple:** M3 Max with 128G memory (24G or more recommended).
*   Specific requirements listed for HiDream, Flux.1, SDXL, and Stable Diffusion 2.x.

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Refer to the [installation documentation](/INSTALL.md) for setup instructions.

## Troubleshooting

*   Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.
*   For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   Consult [this documentation](/OPTIONS.md) for a list of available options.