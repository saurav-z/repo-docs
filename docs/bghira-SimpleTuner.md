# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner** empowers you to easily fine-tune diffusion models with a focus on simplicity and understandability.  Find the original repository [here](https://github.com/bghira/SimpleTuner).

## Key Features

*   **Simplicity:** Designed for ease of use with sensible defaults, reducing the need for extensive configuration.
*   **Versatility:** Supports training on diverse datasets, from small to large image collections.
*   **Cutting-Edge:** Implements proven features for optimal performance, avoiding untested options.
*   **Multi-GPU Training:** Leverage the power of multiple GPUs for faster training.
*   **Efficient Data Handling:**  Image, video, and caption data are cached to hard drive.
*   **Aspect Ratio Support:** Train with varied image and video sizes with aspect bucketing.
*   **LoRA & Full Training:** Supports LoRA and full u-net training for SDXL and other models.
*   **DeepSpeed Integration:** Enables training on memory-constrained systems using DeepSpeed.
*   **Quantization Support:** Utilize low-precision training (NF4/INT8/FP8) to reduce VRAM consumption.
*   **EMA Weighting:** Optional Exponential Moving Average to improve training stability.
*   **S3-Compatible Storage:** Train directly from S3-compatible storage providers (e.g., Cloudflare R2, Wasabi).
*   **ControlNet Training:** Full or LoRA based ControlNet model training.
*   **Mixture of Experts:** Training Mixture of Experts for lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Enhanced convergence and reduced overfitting using Masked loss training.
*   **Prior Regularisation:** Strong prior regularisation support for LyCORIS models.
*   **Webhook Support:** Integrate with services like Discord for training progress updates.
*   **Hugging Face Hub Integration:** Upload trained models and generate model cards.
*   **Extensive Model Support:**  Training support for:
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
    *   Legacy Stable Diffusion models

## Table of Contents

-   [Features](#key-features)
-   [Supported Models](#supported-models)
-   [Hardware Requirements](#hardware-requirements)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Supported Models

SimpleTuner offers training support for a variety of diffusion models, including:

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
*   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)

## Hardware Requirements

Detailed hardware requirements for each model are available in the original README.  Generally, NVIDIA GPUs (3080 and up) are recommended. AMD and Apple silicon are also supported, and training can be performed with various memory configurations, depending on the model and training settings.

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs by setting `SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`). For training loop performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.  A comprehensive list of options can be found in the [OPTIONS documentation](/OPTIONS.md).