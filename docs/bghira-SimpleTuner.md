# SimpleTuner: Simplify and Supercharge Your Image Generation Model Training 🚀

**SimpleTuner** is a user-friendly toolkit designed to streamline the process of fine-tuning image generation models, making cutting-edge features accessible with ease. ([Original Repository](https://github.com/bghira/SimpleTuner))

## Key Features

*   **Versatile Model Support:** Train a wide range of models including HiDream, Flux.1, SDXL, Stable Diffusion 3, and more.
*   **Simplified Configuration:** Designed for ease of use with sensible defaults, reducing the need for extensive tweaking.
*   **Memory Optimization:** Features like aspect bucketing, image/video caching, and DeepSpeed integration to minimize memory consumption.
*   **Advanced Training Techniques:** Supports LoRA, LyCORIS, ControlNet, Mixture of Experts, and masked loss training for enhanced results.
*   **Hardware Flexibility:** Train on a variety of hardware, from 16GB GPUs to multi-GPU setups and Apple silicon.
*   **Integration with Hugging Face Hub:** Easily upload your trained models and leverage the datasets library for compatible datasets.

## Table of Contents

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
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)
-   [Design Philosophy](#design-philosophy)

## Features

### HiDream

*   Custom ControlNet implementation for training via full-rank, LoRA or Lycoris
*   Memory-efficient training for NVIDIA GPUs
*   Optional MoEGate loss augmentation
*   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto`

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Double the training speed of Flux.1 with the new `--fuse_qkv_projections` option
*   ControlNet training via full-rank, LoRA or Lycoris
*   Instruct fine-tuning for the Kontext \[dev] editing model
*   Classifier-free guidance training
*   (optional) T5 attention masked training
*   LoRA or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision`

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text to Video training supported.
*   LyCORIS, PEFT, and full tuning all work as expected

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

*   LyCORIS, PEFT, and full tuning all work as expected

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.

### PixArt Sigma

*   Text encoder training is not supported
*   LyCORIS and full tuning both work as expected
*   ControlNet training is supported
*   Two-stage PixArt training support

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### NVLabs Sana

*   LyCORIS and full tuning both work as expected.

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide to start training.

### Stable Diffusion 3

*   LoRA and full finetuning are supported as usual.
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

*   SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder

### Lumina2

*   LoRA, Lycoris, and full finetuning are supported

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available with example configurations.

### Cosmos2 Predict (Image)

*   Currently, only the text-to-image variant is supported.
*   Lycoris or full-rank tuning are supported

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available with full example configuration and dataset.

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

### NVIDIA

*   Pretty much anything 3080 and up is a safe bet. YMMV.

### AMD

*   LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

### Apple

*   LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.

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

## Design Philosophy

- **Simplicity**: Aiming to have good default settings for most use cases, so less tinkering is required.
- **Versatility**: Designed to handle a wide range of image quantities - from small datasets to extensive collections.
- **Cutting-Edge Features**: Only incorporates features that have proven efficacy, avoiding the addition of untested options.

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/OPTIONS.md).