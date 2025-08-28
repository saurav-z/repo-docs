# SimpleTuner: Simplify Your Image Generation Model Training 🚀

**SimpleTuner** empowers you to easily train various image generation models, offering a straightforward approach for both beginners and experienced users. Check out the original repository [here](https://github.com/bghira/SimpleTuner).

> **Important:** No data is sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

## Key Features

*   **User-Friendly Design:** Prioritizes simplicity and ease of understanding.
*   **Versatile:** Supports a wide range of image quantities, from small datasets to large collections.
*   **Cutting-Edge:** Integrates proven and effective features.
*   **Multi-GPU Training:** Accelerate your training process.
*   **Advanced Techniques:** Includes token-wise dropout, aspect bucketing, and masked loss training.
*   **Memory Optimization:** Features like caching and quantisation to reduce VRAM usage.
*   **Model Support:** Compatible with HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict (Image), and Qwen-Image.
*   **Integration with Hugging Face Hub:** Seamless model uploads and auto-generated model cards.
*   **S3 Storage Support:** Train directly from S3-compatible storage providers.
*   **Webhook Support:** Get real-time training updates to Discord.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Supported Models](#supported-models)
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
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Focus on providing sensible defaults for easy use.
*   **Versatility:** Built to handle diverse dataset sizes effectively.
*   **Innovation:** Includes only proven, impactful features.

## Tutorial

Refer to the [tutorial](/TUTORIAL.md) for detailed guidance, or use the [Quick Start](/documentation/QUICKSTART.md) guide for a fast setup.

For memory optimization, see the [DeepSpeed document](/documentation/DEEPSPEED.md).
For distributed training, see [this guide](/documentation/DISTRIBUTED.md).

## Supported Models

Provides extensive support for many models.

*   **HiDream:** [See Features](#features)
*   **Flux.1:** [See Features](#features)
*   **Wan Video:** [See Features](#features)
*   **LTX Video:** [See Features](#features)
*   **PixArt Sigma:** [See Features](#features)
*   **NVLabs Sana:** [See Features](#features)
*   **Stable Diffusion 3:** [See Features](#features)
*   **Kwai Kolors:** [See Features](#features)
*   **Lumina2:** [See Features](#features)
*   **Cosmos2 Predict (Image):** [See Features](#features)
*   **Qwen-Image:** [See Features](#features)
*   **Legacy Stable Diffusion:** [See Features](#features)

## Hardware Requirements

General guidelines and model-specific requirements are listed.

### NVIDIA

Generally, any 3080 or higher is a good starting point.

### AMD

LoRA and full-rank tuning are verified on a 7900 XTX 24GB and MI300X.  Performance may be lower due to the lack of `xformers`.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory.
*   24GB or greater is recommended.
*   Monitor Pytorch issues for MPS.

### HiDream [dev, full]

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)

### Flux.1 [dev, schnell]

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)
*   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
*   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### Auraflow

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)
*   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
*   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

*   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
*   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
*   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
*   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
*   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

*   16G or better

## Toolkit

Learn about the SimpleTuner toolkit in the [toolkit documentation](/toolkit/README.md).

## Setup

Follow the detailed setup instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logging by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.  For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. A comprehensive list of options is available in the [OPTIONS.md](/OPTIONS.md) file.