# SimpleTuner: Your Simple Solution for Fine-Tuning Diffusion Models

**SimpleTuner** simplifies the process of training diffusion models, offering ease of use and extensive features for a variety of models. ([Original Repo](https://github.com/bghira/SimpleTuner))

## Key Features:

*   **Versatile Model Support:** Train models including Flux, HiDream, Stable Diffusion, and many more.
*   **Memory Optimization:** Features like aspect bucketing, caching, and DeepSpeed integration reduce VRAM usage.
*   **ControlNet & LoRA Training:** Supports ControlNet, LoRA, LyCORIS, and full U-Net training methods for enhanced customization.
*   **Multi-GPU & S3 Training:** Train efficiently on multiple GPUs and directly from S3-compatible storage.
*   **Advanced Techniques:** Includes EMA, quantisation, masked loss, and prior regularisation for improved results.
*   **Seamless Integration:** Integrates with Hugging Face Hub for model upload and dataset loading.
*   **Webhook Support:** Receive training progress updates via webhooks for Discord and other platforms.

## Core Functionality

*   **Design Philosophy:**
    *   **Simplicity**: Prioritising ease of use with good default settings.
    *   **Versatility**: Supports a broad range of image/video datasets, including everything from smaller datasets to massive collections.
    *   **Cutting-Edge Features**: Uses tested features only, and incorporates the latest training methods.

## Model-Specific Support:

*   **HiDream:** Full training support, including custom ControlNet and memory-efficient training.
*   **Flux.1:** Includes support for fast training, ControlNet, and instruction fine-tuning.
*   **Wan Video:** Preliminary text-to-video training.
*   **LTX Video:** Efficient training for less than 16GB VRAM.
*   **PixArt Sigma:** Extensive support including LoRA/full training and ControlNet training.
*   **NVLabs Sana:** Accessible and fast model training with LyCORIS and full tuning.
*   **Stable Diffusion 3:** Includes LoRA, ControlNet, and full finetuning support.
*   **Kwai Kolors:** SDXL-based model with an extended hidden dimension.
*   **Lumina2:** Supports LoRA and full finetuning on a low VRAM model.
*   **Cosmos2 Predict (Image):** Tuning for text-to-image, with full configuration and dataset examples.
*   **Legacy Stable Diffusion Models:** Supports training SD 1.5 and SD 2.x.

## Hardware Requirements:

*   **NVIDIA:** 3080 and up
*   **AMD:** 7900 XTX 24GB and MI300X
*   **Apple:** M3 Max with 128GB (24GB+ recommended)

(Specific requirements for each model are listed in the original README.)

## Setup

*   Find detailed setup instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

*   Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG`.
*   Analyse training performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   Consult the [OPTIONS.md](OPTIONS.md) documentation for a comprehensive list of settings.

## Further Information
*   For information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).
*   Explore the [Quick Start](/documentation/QUICKSTART.md) guide for a faster start.