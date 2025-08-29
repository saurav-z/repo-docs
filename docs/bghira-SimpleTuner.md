# SimpleTuner: Effortless AI Model Training for Everyone 🚀

**SimpleTuner empowers you to easily train and fine-tune cutting-edge AI models, from Stable Diffusion to HiDream, with a focus on simplicity and accessibility.**

[View the original repository on GitHub](https://github.com/bghira/SimpleTuner)

> ℹ️ **Privacy Focused:** SimpleTuner prioritizes your privacy. No data is shared with third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

## Key Features

*   **Simplified Training:** Intuitive design with sensible defaults, reducing the need for complex configurations.
*   **Versatile Compatibility:** Supports a wide range of image and video quantities, from small datasets to massive collections.
*   **Cutting-Edge Models:** Integrates support for the latest and most effective AI models, including:
    *   HiDream
    *   Flux.1
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   Qwen-Image
*   **Performance Optimization:** Features like aspect bucketing, caching, and memory-efficient training techniques (e.g., DeepSpeed integration, quantization) to maximize speed and minimize hardware requirements.
*   **Flexible Training Options:** Supports LoRA, LyCORIS, full finetuning, ControlNet training, Mixture of Experts, and more.
*   **Integration with Hugging Face Hub:** Seamless model uploading and automated model card generation for easy sharing.
*   **Community Support:** Join our community on [Discord](https://discord.gg/CVzhX7ZA) (Terminus Research Group) for help and discussions.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Focus on good defaults for ease of use.
*   **Versatility:** Handles datasets of varying sizes.
*   **Cutting-Edge:** Integrates proven and effective features.

## Tutorial

Start your training journey by exploring the [tutorial](/TUTORIAL.md), which provides vital information.

*   **Quick Start:** For a rapid introduction, see the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Learn how to use 🤗Accelerate to configure Microsoft's DeepSpeed for optimizer state offload. ([DeepSpeed document](/documentation/DEEPSPEED.md))
*   **Multi-node Training:** Optimize for multi-node training using the guide located [here](/documentation/DISTRIBUTED.md).

## Supported Models

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

## Hardware Requirements

The hardware requirements vary depending on the model and training settings. See each model's section for details.

### General Notes
*   **NVIDIA:** Generally, anything from a 3080 and up should work well.
*   **AMD:** Supports LoRA and full-rank tuning (tested on 7900 XTX 24GB and MI300X). May use more memory due to the lack of `xformers`.
*   **Apple:** Tested on M3 Max with 128GB memory. Requires 24GB+ due to lack of memory-efficient attention.
*   See each model's section for more detailed requirements.

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for more details by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`).
For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
Consult [OPTIONS.md](/OPTIONS.md) for a comprehensive list of available options.