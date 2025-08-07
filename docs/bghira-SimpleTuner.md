# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner empowers you to effortlessly train a wide range of diffusion models, from SDXL to HiDream, with a focus on simplicity and accessibility.**

[Link to Original Repository](https://github.com/bghira/SimpleTuner)

**Key Features:**

*   **Broad Model Support:** Train models like SDXL, Stable Diffusion 3, Flux.1, HiDream, and more.
*   **User-Friendly Design:** Prioritizes ease of use with sensible default settings and minimal configuration.
*   **Versatile Training:** Supports various image and video sizes, aspect ratios, and dataset sizes.
*   **Memory Efficiency:** Includes DeepSpeed integration, quantization, and caching to reduce VRAM usage.
*   **Advanced Techniques:** Implements cutting-edge features like TREAD, aspect bucketing, and masked loss training.
*   **Flexible Training Options:** Supports LoRA, LyCORIS, full U-Net training, ControlNet, and Mixture of Experts.
*   **Hugging Face Integration:** Seamlessly upload models and leverage the Hugging Face Hub.
*   **S3-Compatible Storage:** Train directly from cloud storage providers.
*   **Webhook Support:** Receive real-time updates on training progress through webhooks.
*   **Extensive documentation for Troubleshooting:** Comprehensive setup and troubleshooting guides

**Key Features:**

*   **Broad Model Support:** Train models like SDXL, Stable Diffusion 3, Flux.1, HiDream, and more.
*   **User-Friendly Design:** Prioritizes ease of use with sensible default settings and minimal configuration.
*   **Versatile Training:** Supports various image and video sizes, aspect ratios, and dataset sizes.
*   **Memory Efficiency:** Includes DeepSpeed integration, quantization, and caching to reduce VRAM usage.
*   **Advanced Techniques:** Implements cutting-edge features like TREAD, aspect bucketing, and masked loss training.
*   **Flexible Training Options:** Supports LoRA, LyCORIS, full U-Net training, ControlNet, and Mixture of Experts.
*   **Hugging Face Integration:** Seamlessly upload models and leverage the Hugging Face Hub.
*   **S3-Compatible Storage:** Train directly from cloud storage providers.
*   **Webhook Support:** Receive real-time updates on training progress through webhooks.
*   **Extensive documentation for Troubleshooting:** Comprehensive setup and troubleshooting guides

## Table of Contents

*   [Features](#features)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Features

*   [Multi-GPU training](#features)
*   [New token-wise dropout techniques like TREAD](#features) for faster training with Wan 2.1/2.2, and Flux.
*   [Image, video, and caption caching](#features) to speed up training and reduce memory usage
*   [Aspect bucketing](#features): support for a variety of image/video sizes and aspect ratios, enabling widescreen and portrait training.
*   [LoRA and full u-net training for SDXL](#features)
*   [DeepSpeed integration](#features) allowing for training SDXL's full u-net on 12G of VRAM
*   [Quantised NF4/INT8/FP8 LoRA training](#features), using low-precision base model to reduce VRAM consumption.
*   [Optional EMA (Exponential moving average) weight network](#features) to counteract model overfitting and improve training stability.
*   [Train directly from an S3-compatible storage provider](#features), eliminating the requirement for expensive local storage.
*   [Full or LoRA based ControlNet model training](#features) for SDXL, SD 1.x/2.x, and Flux (not ControlLite)
*   [Training Mixture of Experts](#features) for lightweight, high-quality diffusion models
*   [Masked loss training](#features) for superior convergence and reduced overfitting on any model
*   [Prior regularisation](#features) training support for LyCORIS models
*   [Webhook support](#features) for updating eg. Discord channels with your training progress, validations, and errors
*   [Integration with the Hugging Face Hub](https://huggingface.co) for seamless model upload and nice automatically-generated model cards.

### Supported Models

*   [HiDream](#hidream)
*   [Flux.1](#flux1)
*   [Wan Video](#wan-video)
*   [LTX Video](#ltx-video)
*   [PixArt Sigma](#pixart-sigma)
*   [NVLabs Sana](#nvlabs-sana)
*   [Stable Diffusion 3](#stable-diffusion-3)
*   [Kwai Kolors](#kwai-kolors)
*   [Lumina2](#lumina2)
*   [Cosmos2 Predict](#cosmos2-predict)
*   [Qwen-Image](#qwen-image)
*   [Legacy Stable Diffusion](#legacy-stable-diffusion-models)

## Hardware Requirements

*   [NVIDIA](#nvidia)
*   [AMD](#amd)
*   [Apple](#apple)
*   [HiDream](#hidream)
*   [Flux.1](#flux1-dev-schnell)
*   [Auraflow](#auraflow)
*   [SDXL, 1024px](#sdxl-1024px)
*   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

*   [Enable debug logs](#troubleshooting) by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.
*   For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.
*   For a comprehensive list of options available, consult [this documentation](/OPTIONS.md).