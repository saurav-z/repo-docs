# SimpleTuner: Train Cutting-Edge AI Models with Ease 🚀

**SimpleTuner** empowers you to easily train a wide range of AI models, with a focus on simplicity and cutting-edge features. [Visit the original repo](https://github.com/bghira/SimpleTuner) for more details and to contribute.

## Key Features

*   **User-Friendly Design:** SimpleTuner prioritizes ease of use with sensible default settings, minimizing the need for complex configurations.
*   **Versatile Training:** Supports diverse datasets and image quantities, from small collections to massive, multi-billion sample datasets.
*   **State-of-the-Art Capabilities:** Implements proven, cutting-edge features for optimized training performance and model quality.
*   **Multi-GPU Training:** Train models faster with support for multiple GPUs.
*   **Advanced Techniques:** Includes new token-wise dropout techniques like TREAD for faster training.
*   **Caching:** Efficient caching of image, video, and caption data to accelerate training and reduce memory usage.
*   **Aspect Ratio Support:** Train with diverse image/video sizes and aspect ratios for flexibility.
*   **LoRA/LyCORIS & Full U-Net Training:** Supports a variety of model types, including LoRA/LyCORIS training for models like PixArt, SDXL, SD3, and SD 2.x, and even full u-net training for SDXL.
*   **Memory Optimization:** Enables training on GPUs with as little as 16GB VRAM through techniques like DeepSpeed integration and quantization.
*   **Quantization:** Utilize quantized NF4/INT8/FP8 LoRA training for reducing VRAM consumption.
*   **EMA (Exponential Moving Average):** Optional EMA weight network to improve stability and mitigate overfitting.
*   **S3 Storage Support:** Directly train from S3-compatible storage providers, reducing the need for local storage.
*   **ControlNet Training:** Support for full or LoRA-based ControlNet model training for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts Training:** Train using Mixture of Experts to create lightweight and high-quality diffusion models.
*   **Masked Loss Training:** Implement masked loss training for improved convergence and reduce overfitting.
*   **Prior Regularization:** Supports strong prior regularization training.
*   **Webhook Support:** Integrate webhooks for receiving real-time updates on training progress and errors.
*   **Hugging Face Hub Integration:** Seamlessly upload and share trained models on the Hugging Face Hub.

## Supported Models

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
*   Legacy Stable Diffusion

## Hardware Requirements

*   **NVIDIA:** 3080 and up recommended
*   **AMD:** LoRA and full-rank tuning verified working on a 7900 XTX 24GB and MI300X.
*   **Apple:** Tested to work on an M3 Max with 128GB.
*   **Model Specific Requirements:** Refer to the documentation for specific model hardware recommendations.

## Sections
*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity**: Good defaults for most use cases.
*   **Versatility**: Supports a wide range of image quantities.
*   **Cutting-Edge Features**: Incorporates proven features.

## Tutorial

Begin your training journey with the comprehensive [tutorial](/TUTORIAL.md).

For a quick start, refer to the [Quick Start](/documentation/QUICKSTART.md) guide.

## Toolkit

Explore the capabilities of the [toolkit](/toolkit/README.md) to assist with model training and management.

## Setup

Follow the detailed [installation documentation](/INSTALL.md) to get started.

## Troubleshooting

Enable debugging logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG`.
Analyze training loop performance by setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For available options, see [this documentation](/OPTIONS.md).