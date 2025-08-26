# SimpleTuner: Your Gateway to Accessible and Powerful Diffusion Model Training

**Unlock the potential of diffusion models with SimpleTuner, a user-friendly and versatile toolkit designed for both beginners and experts.** (Check out the original repo [here](https://github.com/bghira/SimpleTuner)!)

## Key Features

*   **Simplified Design:** Easy to understand code with sensible defaults, reducing the need for extensive configuration.
*   **Broad Compatibility:** Supports a wide range of image quantities and aspect ratios, from small datasets to massive collections.
*   **Cutting-Edge Techniques:** Implements proven and effective features, avoiding untested options.
*   **Multi-GPU Support:** Leverage the power of multiple GPUs for faster training.
*   **Advanced Training Techniques:** Includes token-wise dropout (TREAD), aspect bucketing, refiner LoRA, and EMA for improved performance and stability.
*   **VRAM Optimization:** Features like DeepSpeed integration, quantization (NF4/INT8/FP8), and S3-compatible storage support to reduce VRAM consumption.
*   **Hugging Face Integration:** Seamless model upload and model card generation via the Hugging Face Hub.
*   **Extensive Model Support:** Comprehensive training integration for:
    *   HiDream
    *   Flux.1
    *   Wan Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   Qwen-Image
    *   Legacy Stable Diffusion (SD 1.5 & 2.x)

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

*   **Simplicity:** Designed for ease of use with sensible defaults.
*   **Versatility:** Supports a wide range of datasets and image sizes.
*   **Proven Features:** Focuses on established and effective training techniques.

## Tutorial

Get started with the [tutorial](/TUTORIAL.md) and [Quick Start](/documentation/QUICKSTART.md) guide to learn more.

## Hardware Requirements

The hardware requirements vary depending on the specific model and configuration.  Refer to the documentation for specific GPU recommendations:

*   [NVIDIA](#nvidia)
*   [AMD](#amd)
*   [Apple](#apple)
*   [HiDream](#hidream)
*   [Flux.1](#flux1)
*   [Auraflow](#auraflow)
*   [SDXL](#sdxl-1024px)
*   [Stable Diffusion 2.x](#stable-diffusion-2x-768px)

## Toolkit

For information on the SimpleTuner toolkit, check out the [toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment. For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. A comprehensive list of options is in the [options documentation](/OPTIONS.md).