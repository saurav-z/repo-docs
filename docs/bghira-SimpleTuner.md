# SimpleTuner: Effortless Fine-tuning for Cutting-Edge AI Models

**SimpleTuner** is a user-friendly toolkit designed for straightforward and efficient fine-tuning of various AI models, enabling you to easily train and customize models for your specific needs.  You can find the original repo [here](https://github.com/bghira/SimpleTuner).

> ℹ️ No data is sent to any third parties except through opt-in flag `report_to`, `push_to_hub`, or webhooks which must be manually configured.

## Key Features

*   **User-Friendly Design:** Simplifies the fine-tuning process with intuitive settings and good defaults.
*   **Versatile:** Supports a wide range of image and video dataset sizes and aspect ratios.
*   **Cutting-Edge Techniques:** Incorporates proven methods to improve performance and stability, including LoRA, LyCORIS, and ControlNet training.
*   **Broad Model Support:** Compatible with a variety of models, including SDXL, SD3, HiDream, Flux.1, and more.
*   **Memory Optimization:** Features like quantisation, DeepSpeed integration, and caching to enable training on GPUs with limited VRAM.
*   **Seamless Integration:**  Supports direct training from S3-compatible storage and integration with the Hugging Face Hub.
*   **Advanced Training Options:** Includes masked loss, prior regularisation, and webhook support for detailed monitoring.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorials & Quick Start](#tutorials--quick-start)
-   [Features](#features)
    -   [Model Support](#model-support)
    -   [Training Techniques](#training-techniques)
    -   [Integration & Optimizations](#integration--optimizations)
-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [Model Specific Requirements](#model-specific-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity:** Focus on ease of use with sensible defaults.
-   **Versatility:** Adaptable to diverse datasets and image sizes.
-   **Performance:** Employs the latest, well-validated methods.

## Tutorials & Quick Start

*   Comprehensive [Tutorial](/TUTORIAL.md): Learn the fundamentals of fine-tuning with SimpleTuner.
*   [Quick Start](/documentation/QUICKSTART.md): Get started with a streamlined setup.
*   [DeepSpeed Guide](/documentation/DEEPSPEED.md): Optimize for memory-constrained systems using DeepSpeed.
*   [Distributed Training](/documentation/DISTRIBUTED.md): Configure multi-node training for large datasets.

## Features

### Model Support
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
*   [Legacy Stable Diffusion](#legacy-stable-diffusion-models)

### Training Techniques

*   Multi-GPU training
*   New token-wise dropout techniques like [TREAD](/documentation/TREAD.md)
*   Aspect bucketing
*   Refiner LoRA or full u-net training for SDXL
*   DeepSpeed integration for memory-constrained training
*   Quantised NF4/INT8/FP8 LoRA training
*   Optional EMA (Exponential moving average)
*   [ControlNet model training](/documentation/CONTROLNET.md)
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md)
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss)
*   Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data)

### Integration & Optimizations

*   Image, video, and caption features (embeds) are cached to the hard drive for faster training.
*   Train directly from an S3-compatible storage provider (Cloudflare R2, Wasabi S3).
*   Integration with the [Hugging Face Hub](https://huggingface.co) for model uploads and model cards.
*   Webhook support for monitoring training progress and errors.
*   Use the [datasets library](/documentation/data_presets/preset_subjects200k.md) ([more info](/documentation/HUGGINGFACE_DATASETS.md)) to load compatible datasets directly from the hub.

## Hardware Requirements

### NVIDIA

NVIDIA GPUs (3080 and up) are recommended.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

### Apple

LoRA and full-rank tuning tested to work on an M3 Max with 128G memory. Requires at least 24G memory for machine learning tasks.

### Model Specific Requirements

See the respective model sections below for model-specific hardware recommendations.
*   [HiDream](#hidream)
*   [Flux.1](#flux1-dev-schnell)
*   [SDXL](#sdxl-1024px)
*   [Stable Diffusion (Legacy)](#stable-diffusion-2x-768px)

## Toolkit

Refer to the [toolkit documentation](/toolkit/README.md) for details about the tools provided.

## Setup

Detailed setup instructions can be found in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for more detailed insights by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file. For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. Consult [this documentation](/OPTIONS.md) for available options.