# SimpleTuner: Train Cutting-Edge AI Models with Ease 🚀

> SimpleTuner empowers you to train a variety of AI models with simplicity, versatility, and efficiency, focusing on ease of use and cutting-edge features.

This project focuses on streamlined AI model training, providing a user-friendly experience with a focus on clear code and practical functionality. Contributions are welcome! Explore the original repository on [GitHub](https://github.com/bghira/SimpleTuner).

**Key Features:**

*   **Wide Model Support:** Train popular models like Stable Diffusion (SDXL, SD3), HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Kwai Kolors, Lumina2, Cosmos2 Predict, and Qwen-Image.
*   **Simplified Training:** Optimized for ease of use with good default settings and supports image, video and caption features.
*   **Versatile:** Handles diverse datasets, from small to massive, accommodating a wide range of image and video sizes.
*   **Memory Efficiency:** Features techniques like aspect bucketing, caching, quantisation, and DeepSpeed integration to reduce VRAM consumption.
*   **Cutting-Edge Techniques:** Includes LoRA/LyCORIS training, EMA, ControlNet training, Mixture of Experts (MoE), masked loss training, and prior regularization.
*   **Hardware Flexibility:** Works with a variety of GPUs (NVIDIA, AMD), CPUs, and even Apple silicon.
*   **Hugging Face Integration:** Seamless model uploads and dataset loading from the Hugging Face Hub.
*   **Webhook Support:** Receive training progress updates and notifications.
*   **Performance Optimization:** Utilize techniques like TREAD dropout, Flash Attention, and fuse QKV projections for faster training.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorials](#tutorials)
-   [Features](#features)
-   [Hardware Requirements](#hardware-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizes easy-to-understand code and sensible default settings for a streamlined experience.
*   **Versatility:** Supports a broad range of datasets and image/video sizes.
*   **Cutting-Edge:** Integrates proven and effective features while avoiding untested options.

## Tutorials

*   **Main Tutorial:** The main tutorial is available at [the tutorial](/TUTORIAL.md), containing all of the essential information you need.
*   **Quick Start Guide:** For a fast start, check out the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** For memory-constrained systems, use the [DeepSpeed document](/documentation/DEEPSPEED.md) guide.
*   **Distributed Training:** For multi-node training, see the [distributed training guide](/documentation/DISTRIBUTED.md)

## Features

*(See Key Features section above for a condensed summary, or consult the original README for more detailed model-specific information).*

## Hardware Requirements

*(See original README for hardware requirement details, which is comprehensive but redundant)*

## Toolkit

For information about the associated toolkit, see the [toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`) file.

For detailed training loop performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For available options, consult [this documentation](/OPTIONS.md).