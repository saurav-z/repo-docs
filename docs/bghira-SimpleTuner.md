# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease 🚀

> **SimpleTuner empowers you to train various diffusion models efficiently, with a focus on simplicity and cutting-edge features.**

SimpleTuner is designed for ease of use, providing excellent default settings and a streamlined experience for both beginners and experienced users. This repository serves as a collaborative academic exercise, with contributions welcome. Note that no data is sent to third parties except through opt-in flags (`report_to`, `push_to_hub`) or manually configured webhooks.

**Explore the SimpleTuner documentation and start training your own models:** [https://github.com/bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)

Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for questions and support!

## Key Features

*   **Versatile Model Support:** Train a wide range of models, including:
    *   HiDream
    *   Flux.1
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3.0
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict
    *   Qwen-Image
    *   Legacy Stable Diffusion (1.x, 2.x)
*   **Memory-Efficient Training:** Utilizes techniques like caching, aspect bucketing, and quantisation to minimize VRAM usage, enabling training on GPUs with as little as 16GB VRAM.
*   **Multi-GPU Training:** Supports multi-GPU training for faster model training.
*   **Advanced Training Techniques:** Incorporates cutting-edge features such as TREAD (token-wise dropout), ControlNet training, Mixture of Experts, Masked Loss training, and prior regularisation.
*   **DeepSpeed Integration:** Offers DeepSpeed integration for training large models on systems with limited VRAM.
*   **Seamless Integration:** Supports training directly from S3-compatible storage, Hugging Face Hub integration (datasets library), and webhooks for monitoring training progress.
*   **Flexible Configuration:** Provides a wide range of options for customisation, allowing you to tailor the training process to your specific needs.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorials & Quick Starts](#tutorials-quick-starts)
*   [Hardware Requirements](#hardware-requirements)
*   [Scripts](#scripts)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Good default settings to minimize the need for extensive configuration.
*   **Versatility:** Designed to handle datasets of various sizes.
*   **Cutting-Edge:** Incorporates only proven, effective features.

## Tutorials & Quick Starts

*   **Comprehensive Guide:** Explore the full [tutorial](/TUTORIAL.md) for detailed information.
*   **Quick Start:** Get started quickly with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Learn about [DeepSpeed](/documentation/DEEPSPEED.md) for memory-constrained systems.
*   **Multi-Node Training:** [Guide](/documentation/DISTRIBUTED.md) for multi-node distributed training.
*   Quickstarts are also available for specific models, as mentioned under the "Features" section, above.

## Hardware Requirements

Requirements vary depending on the model and configuration.
*   **NVIDIA:** Generally, a 3080 or higher is recommended.
*   **AMD:** LoRA and full-rank tuning verified on a 7900 XTX 24GB and MI300X.
*   **Apple:** Works on M3 Max with 128GB memory, but requires 24GB+ for ML.
*   Specific model requirements are detailed in the original README.

## Scripts

The original README contains a list of scripts.

## Toolkit

For information on the associated toolkit, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for detailed insights by setting `SIMPLETUNER_LOG_LEVEL=DEBUG` or `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` in your environment.
For a list of options, consult [this documentation](/OPTIONS.md).