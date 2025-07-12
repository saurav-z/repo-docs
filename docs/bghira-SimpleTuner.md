# SimpleTuner: Simplify Your AI Model Training 📈

**Unlock the power of AI image generation with SimpleTuner, a user-friendly toolkit for training cutting-edge diffusion models.** Learn more at the [original repository](https://github.com/bghira/SimpleTuner).

**Important:** *Always back up your training data, as scripts can potentially modify it.*

SimpleTuner focuses on ease of use, providing intuitive tools and excellent default settings for a variety of training scenarios. Join our community on [Discord](https://discord.com/invite/eq3cAMZtCC) via Terminus Research Group.

**Key Features:**

*   **Versatile Model Support:** Train on popular models like SDXL, Stable Diffusion 3, PixArt Sigma, and more.
*   **Memory Efficiency:** Utilize techniques like caching, aspect bucketing, and DeepSpeed integration to train on GPUs with limited VRAM (down to 16GB).
*   **Advanced Techniques:** Benefit from features like LoRA/LyCORIS, ControlNet training, Mixture of Experts, and masked loss training for improved results.
*   **Hardware Agnostic:** Train on NVIDIA, AMD, and Apple silicon hardware.
*   **Integration:** Seamlessly upload trained models to the Hugging Face Hub.

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
    -   [Legacy Stable Diffusion](#legacy-stable-diffusion-models)
-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Auraflow](#auraflow)
    -   [SDXL](#sdxl-1024px)
    -   [Stable Diffusion 2.x](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Good defaults for minimal configuration.
*   **Versatility:** Handles diverse image quantities and aspect ratios.
*   **Performance:** Focus on proven, effective features.

## Tutorial

Begin by exploring this README, then proceed to the [tutorial](/TUTORIAL.md).
For a quick start, see the [Quick Start](/documentation/QUICKSTART.md) guide.

## Features

*   **Multi-GPU training**
*   **Caching:** Image, video, and caption caching for faster training and reduced memory usage.
*   **Aspect bucketing:** Support for varied image and video sizes.
*   **LoRA/LyCORIS:** Training support for PixArt, SDXL, SD3, and SD 2.x, using less than 16G VRAM.
*   **DeepSpeed Integration:** Train SDXL's full u-net with DeepSpeed (see [DeepSpeed document](/documentation/DEEPSPEED.md))
*   **Quantization:** NF4/INT8/FP8 LoRA training for reduced VRAM consumption.
*   **EMA:** Exponential moving average weight network for stability.
*   **S3 support:** Train directly from S3-compatible storage.
*   **ControlNet:** Full or LoRA based [ControlNet model training](/documentation/CONTROLNET.md)
*   **Mixture of Experts:** Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Superior convergence and reduced overfitting (see [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss))
*   **Prior Regularisation:** Strong training support for LyCORIS models (see [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data))
*   **Webhooks:** Track progress via webhooks (e.g., Discord).
*   **Hugging Face Hub Integration:** Model upload and auto-generated model cards.

### HiDream

*   Custom ControlNet implementation
*   Memory-efficient training for NVIDIA GPUs.
*   MoEGate loss augmentation (optional)
*   Lycoris or full tuning via DeepSpeed ZeRO
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings
*   Quantise Llama LLM using `--text_encoder_4_precision` set to `int4-quanto` or `int8-quanto` to run on 24G cards.

See [HiDream Hardware Requirements](#hidream) or the [HiDream Quickstart](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   **Optimized Speed:** Double Flux.1 training speed with `--fuse_qkv_projections`.
*   ControlNet training via full-rank, LoRA, or Lycoris
*   Instruct fine-tuning for the Kontext \[dev] editing model implementation
*   Classifier-free guidance training (optional)
*   T5 attention masked training (optional)
*   LoRA or full tuning via DeepSpeed ZeRO
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-torchao` for major memory savings

See [Flux.1 Hardware Requirements](#flux1) or the [Flux.1 Quickstart](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text to Video training is supported.
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

*   LoRA and full finetuning are supported
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

*   An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder.

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

## Hardware Requirements

### NVIDIA

Pretty much anything 3080 and up is a safe bet. YMMV.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

Lacking `xformers`, it will use more memory than Nvidia equivalent hardware.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory.
  - You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.
  - Subscribing to Pytorch issues for MPS is probably a good idea, as random bugs will make training stop working.

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

Learn more about the [SimpleTuner toolkit](/toolkit/README.md).

## Setup

Detailed setup instructions are in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`).
For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
For a full list of options, consult [OPTIONS.md](/OPTIONS.md).