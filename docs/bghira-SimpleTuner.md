# SimpleTuner: Your Simplified Guide to Diffusion Model Training 💹

**SimpleTuner provides a streamlined, user-friendly approach to training various diffusion models, prioritizing simplicity and ease of understanding.** Explore the [original repo](https://github.com/bghira/SimpleTuner) for more details and contributions.

**Key Features:**

*   **Simplified Design:** Focuses on intuitive defaults and minimal configuration.
*   **Versatile:** Supports a wide range of datasets, from small to large-scale.
*   **Cutting-Edge:** Integrates only proven, effective features.
*   **Multi-GPU Training:** Enhances training speed and efficiency.
*   **Caching:** Image, video, and caption caching for faster training.
*   **Aspect Bucketing:** Supports diverse image and video aspect ratios.
*   **LoRA/LyCORIS Training:** Efficient training for various models with reduced VRAM usage.
*   **DeepSpeed Integration:** Enables training large models on limited VRAM.
*   **Quantization:** Supports NF4/INT8/FP8 LoRA for reduced memory consumption.
*   **EMA Weighting:** Improves model stability and prevents overfitting.
*   **S3 Storage Support:** Train directly from S3-compatible storage.
*   **ControlNet Support:** Full or LoRA-based ControlNet training for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts:** Enables training for lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Enhances convergence and reduces overfitting.
*   **Prior Regularization:** Supports LyCORIS models for better performance.
*   **Webhook Integration:** Real-time training progress updates via webhooks (e.g., Discord).
*   **Hugging Face Hub Integration:** Seamless model upload and model card generation.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Wan Video](#wan-video)
    -   [LTX Video](#ltx-video)
    -   [PixArt Sigma](#pixart-sigma)
    -   [NVLabs Sana](#nvlabs-sana)
    -   [Stable Diffusion 3](#stable-diffusion-3)
    -   [Kwai Kolors](#kwai-kolors)
    -   [Lumina2](#lumina2)
    -   [Cosmos2 Predict (Image)](#cosmos2-predict-image)
    -   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)
-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Auraflow](#auraflow)
    -   [SDXL, 1024px](#sdxl-1024px)
    -   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity**: Designed with good default settings for ease of use.
-   **Versatility**: Handles a wide range of image and video datasets.
-   **Cutting-Edge**: Incorporates features with proven efficacy.

## Tutorial

Refer to the [Tutorial](/TUTORIAL.md) for in-depth information before starting.
For a quick start, consult the [Quick Start](/documentation/QUICKSTART.md) guide.
For memory optimization, see the [DeepSpeed document](/documentation/DEEPSPEED.md).
For multi-node training, refer to [this guide](/documentation/DISTRIBUTED.md).

---

## Features

*(See the "Key Features" section above for a concise overview of the features.)*

### HiDream

-   Custom ControlNet implementation.
-   Memory-efficient training.
-   Lycoris or full tuning via DeepSpeed ZeRO.
-   Quantise the base model for major memory savings.
-   Quantise Llama LLM to run on 24G cards.

See the [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

-   Faster training with `--fuse_qkv_projections`.
-   ControlNet training.
-   Instruct fine-tuning for the Kontext editing model.
-   Classifier-free guidance training.
-   (optional) T5 attention masked training.
-   LoRA or full tuning via DeepSpeed ZeRO.
-   Quantise the base model for major memory savings.

See the [hardware requirements](#flux1) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

-   Text-to-Video training support.
-   LyCORIS, PEFT, and full tuning are supported.

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md).

### LTX Video

-   LyCORIS, PEFT, and full tuning are supported.

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md).

### PixArt Sigma

-   LyCORIS and full tuning supported.
-   ControlNet training support.
-   Two-stage PixArt training support.

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md).

### NVLabs Sana

-   LyCORIS and full tuning supported.

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md).

### Stable Diffusion 3

-   LoRA and full finetuning are supported as usual.
-   ControlNet training via full-rank, PEFT LoRA, or Lycoris.

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).

### Kwai Kolors

An SDXL-based model with ChatGLM.

### Lumina2

A 2B parameter flow-matching model.

See the [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md).

### Cosmos2 Predict (Image)

A 2B / 14B parameter model that can do video as well as text-to-image.

See the [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md).

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

### NVIDIA

*(See the original documentation for further details.)*

### AMD

*(See the original documentation for further details.)*

### Apple

*(See the original documentation for further details.)*

### HiDream [dev, full]

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)

### Flux.1 [dev, schnell]

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)
-   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
-   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### Auraflow

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)
-   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
-   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

-   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
-   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
-   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
-   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
-   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

-   16G or better

## Toolkit

Learn more about the associated toolkit in the [toolkit documentation](/toolkit/README.md).

## Setup

Follow the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs via `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`).
For training loop performance, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
For available options, consult [this documentation](/OPTIONS.md).