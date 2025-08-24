# SimpleTuner: Train Cutting-Edge AI Models with Ease 🚀

**SimpleTuner** is a user-friendly toolkit designed for training a wide range of AI models with a focus on simplicity and ease of use. [View the Original Repository](https://github.com/bghira/SimpleTuner)

> ℹ️ No data is sent to any third parties except through opt-in features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

**Key Features:**

*   **Simplified Training**: Get started quickly with sensible default settings.
*   **Versatile Compatibility**: Supports diverse image quantities and aspect ratios.
*   **Cutting-Edge Models**: Train the latest models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, and more.
*   **Hardware Optimization**: Supports multi-GPU training, DeepSpeed integration, and low-precision training (NF4/INT8/FP8) for reduced VRAM consumption.
*   **Advanced Techniques**: Includes support for TREAD, aspect bucketing, LoRA/LyCORIS training, EMA, S3-compatible storage, ControlNet, Mixture of Experts, and more.
*   **Hugging Face Hub Integration**: Seamlessly upload and share your trained models.
*   **Comprehensive Documentation**: Detailed guides and quickstarts for each model.
*   **Webhook Support**: Integrate with your training progress with webhooks.

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

*   **Simplicity**: Aiming to have good default settings for most use cases, so less tinkering is required.
*   **Versatility**: Designed to handle a wide range of image quantities - from small datasets to extensive collections.
*   **Cutting-Edge Features**: Only incorporates features that have proven efficacy, avoiding the addition of untested options.

## Tutorial

Explore the [tutorial](/TUTORIAL.md) for in-depth information and the [Quick Start](/documentation/QUICKSTART.md) guide for a rapid introduction. Memory-constrained systems can benefit from the [DeepSpeed document](/documentation/DEEPSPEED.md). For multi-node training, see the [DISTRIBUTED guide](/documentation/DISTRIBUTED.md).

---

## Features

*   **Multi-GPU Training**
*   **New Token-wise Dropout Techniques** like [TREAD](/documentation/TREAD.md) for accelerated training.
*   **Efficient Data Handling**: Image, video, and caption features are cached to the hard drive, optimizing speed and memory.
*   **Aspect Ratio Support**: Training for various image and video sizes, including widescreen and portrait.
*   **Refiner/Full U-Net Training**: For SDXL.
*   **Memory Efficiency**: Most models trainable on 24GB or 16GB GPUs.
    *   LoRA/LyCORIS training for PixArt, SDXL, SD3, and SD 2.x that uses less than 16G VRAM
*   **DeepSpeed Integration**: Enables training SDXL's full u-net on 12GB of VRAM.
*   **Quantization**: NF4/INT8/FP8 LoRA training to reduce VRAM usage.
*   **EMA**: Optional Exponential Moving Average weight network to improve training stability.
*   **S3 Storage**: Train directly from S3-compatible providers.
*   **ControlNet Training**: Full or LoRA based ControlNet model training for SDXL, SD 1.x/2.x, and Flux
*   **Mixture of Experts**: Training for lightweight, high-quality diffusion models.
*   **Masked Loss**: Support for superior convergence and reduced overfitting.
*   **Prior Regularization**: Strong support for LyCORIS models.
*   **Webhooks**: Update your training progress, validations, and errors.
*   **Hugging Face Hub**: Seamless model upload and model card creation.
    *   Utilize the [datasets library](/documentation/data_presets/preset_subjects200k.md) to load compatible datasets directly from the hub.

### HiDream

*   Custom ControlNet implementation for training via full-rank, LoRA or Lycoris
*   Memory-efficient training for NVIDIA GPUs (AMD support is planned)
*   Optional MoEGate loss augmentation
*   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings
*   Quantise Llama LLM using `--text_encoder_4_precision` set to `int4-quanto` or `int8-quanto` to run on 24G cards.

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Double the training speed with `--fuse_qkv_projections` option, taking advantage of Flash Attention 3 on Hopper systems
*   ControlNet training via full-rank, LoRA or Lycoris
*   Instruct fine-tuning for the Kontext editing model.
*   Classifier-free guidance training
*   (optional) T5 attention masked training for superior fine details and generalisation capabilities
*   LoRA or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-torchao` for major memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text to Video training is supported.
*   Image to Video training is not yet supported.
*   Text encoder training is not supported.
*   VAE training is not supported.
*   LyCORIS, PEFT, and full tuning all work as expected
*   ControlNet training is not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

*   Text encoder training is not supported
*   VAE training is not supported
*   LyCORIS, PEFT, and full tuning all work as expected
*   ControlNet training is not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.

### PixArt Sigma

*   Text encoder training is not supported
*   LyCORIS and full tuning both work as expected
*   ControlNet training is supported for full and PEFT LoRA training
*   [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support (see: [MIXTURE_OF_EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md))

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### NVLabs Sana

*   LyCORIS and full tuning both work as expected.
*   Text encoder training is not supported.
*   PEFT Standard LoRA is not supported.
*   ControlNet training is not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide to start training.

### Stable Diffusion 3

*   LoRA and full finetuning are supported as usual.
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris
*   Certain features such as segmented timestep selection and Compel long prompt weighting are not yet supported.
*   Parameters have been optimised to get the best results, validated through from-scratch training of SD3 models

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

*   SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder
*   Kolors support is almost as deep as SDXL, minus ControlNet training support.

### Lumina2

*   LoRA, Lycoris, and full finetuning are supported
*   ControlNet training is not yet supported

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available with example configurations.

### Cosmos2 Predict (Image)

*   Currently, only the text-to-image variant is supported.
*   Lycoris or full-rank tuning are supported, but PEFT LoRAs are currently not.
*   ControlNet training is not yet supported.

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available with full example configuration and dataset.

### Qwen-Image

*   Lycoris, LoRA, and full-rank training are all supported, with full-rank training requiring H200 or better with DeepSpeed
*   ControlNet training is not yet supported.

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available with example configuration and dataset, as well as general training/configuration tips.

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

### NVIDIA

A 3080 or higher is generally recommended.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.
  * You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.

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

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Follow the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`) file. Use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` for performance analysis.  For available options, consult [this documentation](/OPTIONS.md).