# SimpleTuner: Train Cutting-Edge AI Models with Ease 🚀

SimpleTuner simplifies the process of training AI models, offering a user-friendly experience with a focus on code clarity. **This repository is designed for academic collaboration and welcomes contributions from the community.** Explore the power of fine-tuning with SimpleTuner and unlock new possibilities in AI!

[View the original repository on GitHub](https://github.com/bghira/SimpleTuner)

*   ⚠️ **Privacy Focused:** No data is sent to third parties unless explicitly enabled via the `report_to` flag, `push_to_hub`, or manually configured webhooks.

## Key Features

*   **Simplified Training:** Easy-to-understand codebase for a streamlined training experience.
*   **Versatile Support:** Handles various image and video sizes and aspect ratios, from small datasets to extensive collections.
*   **Multi-GPU Training:** Leverage multiple GPUs for faster training.
*   **Optimized Data Handling:** Caches image, video, and caption features for improved speed and reduced memory consumption.
*   **Aspect Ratio Bucketing:** Train with various image/video sizes and aspect ratios, including widescreen and portrait.
*   **SDXL Refinement:** Support for LoRA or full U-Net training for SDXL.
*   **Low VRAM Training:** Train most models, including LoRA/LyCORIS for PixArt, SDXL, SD3, and SD 2.x, on 24GB or even 16GB GPUs.
*   **DeepSpeed Integration:** Enables training SDXL's full U-Net on 12GB of VRAM.
*   **Quantization:** Utilizes quantised NF4/INT8/FP8 LoRA training to reduce VRAM usage.
*   **EMA Support:** Optional Exponential Moving Average (EMA) for enhanced model stability and to combat overfitting.
*   **S3 Integration:** Train directly from S3-compatible storage providers.
*   **ControlNet Training:** Full or LoRA based ControlNet training for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts (MoE):** Training Mixture of Experts for lightweight, high-quality diffusion models.
*   **Masked Loss Training:**  For superior convergence and reduced overfitting on any model.
*   **Prior Regularization:** Strong prior regularisation training support for LyCORIS models
*   **Webhook Support:** Integrates with webhooks for monitoring and progress updates.
*   **Hugging Face Hub Integration:** Seamlessly upload and manage your models on Hugging Face Hub.
*   **HiDream Support:** Full training support for HiDream, including custom ControlNet implementation and memory-efficient training.
*   **Flux.1 Support:** Accelerated training with Flash Attention 3 and other features.
*   **Wan Video Support:** Preliminary text-to-video training integration.
*   **LTX Video Support:** Efficient training on less than 16GB.
*   **PixArt Sigma Support:** Extensive integration, with LoRA and full training options.
*   **NVLabs Sana Support:** Lightweight and fast model training integration.
*   **Stable Diffusion 3 Support:** LoRA, full finetuning, and ControlNet options.
*   **Kwai Kolors Support:**  An SDXL-based model with enhanced prompt embeddings.
*   **Lumina2 Support:**  LoRA, Lycoris, and full finetuning options.
*   **Cosmos2 Predict Support:** Supports text-to-image variant with Lycoris or full-rank tuning options.
*   **Legacy SD Support:** Compatible with older Stable Diffusion models (1.5 and 2.x).

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
-   [Hardware Requirements](#hardware-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Easy-to-use, with good default settings for most use cases.
*   **Versatility:** Designed to handle various image/video sizes, aspect ratios and dataset sizes.
*   **Cutting-Edge:** Focuses on features that have proven effectiveness, avoiding untested options.

## Tutorial

Start your journey with the full [Tutorial](/TUTORIAL.md) for vital information, or the [Quick Start](/documentation/QUICKSTART.md) guide for a rapid setup.

## Hardware Requirements

*   **NVIDIA:** 3080 and up recommended.
*   **AMD:**  LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.
*   **Apple:** Tested on an M3 Max with 128GB memory, requires 24GB or greater machine.

Specific requirements for different models (HiDream, Flux.1, SDXL, and Stable Diffusion) are detailed in the original README.

## Toolkit

Explore the associated toolkit with SimpleTuner by referring to the [toolkit documentation](/toolkit/README.md).

## Setup

Find detailed installation instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment.

For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

Consult [OPTIONS.md](/OPTIONS.md) for a comprehensive list of available options.