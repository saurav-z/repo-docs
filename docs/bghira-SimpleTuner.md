# SimpleTuner: Train Diffusion Models with Ease 💹

**SimpleTuner is a user-friendly and versatile toolkit designed to simplify the process of training diffusion models.** This repository offers a streamlined approach, making the code easy to understand and use. [Explore the original repository here](https://github.com/bghira/SimpleTuner).

**Key Features:**

*   **Simplicity:** Designed for ease of use with sensible default settings.
*   **Versatility:** Supports a wide range of image quantities, from small to large datasets.
*   **Multi-GPU Training:** Enables faster training with multiple GPUs.
*   **Aspect Bucketing:** Supports diverse image/video sizes and aspect ratios.
*   **Comprehensive Model Support:**
    *   HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Cosmos2, and Lumina2 models.
*   **Memory Optimization:** Caches image, video, and caption features for reduced memory consumption.
*   **Advanced Techniques:** Includes LoRA, LyCORIS, and controlnet training with optional EMA.
*   **DeepSpeed Integration:** Offers DeepSpeed integration for training large models on limited VRAM.
*   **Quantization Support:** Supports quantisation (NF4/INT8/FP8) for reduced VRAM usage.
*   **Cloud Storage Integration:** Train directly from S3-compatible storage.
*   **Hugging Face Hub Integration:** Seamlessly upload and manage your models on the Hugging Face Hub.
*   **Webhook Support:** Monitor training progress with webhook integration.

## Table of Contents

-   [Features](#features)
-   [Model Support](#model-support)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux-1)
    -   [Wan Video](#wan-video)
    -   [LTX Video](#ltx-video)
    -   [PixArt Sigma](#pixart-sigma)
    -   [NVLabs Sana](#nvlabs-sana)
    -   [Stable Diffusion 3](#stable-diffusion-3)
    -   [Kwai Kolors](#kwai-kolors)
    -   [Lumina2](#lumina2)
    -   [Cosmos2 Predict (Image)](#cosmos2-predict-image)
-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream [dev, full]](#hidream-dev-full)
    -   [Flux.1 [dev, schnell]](#flux1-dev-schnell)
    -   [Auraflow](#auraflow)
    -   [SDXL, 1024px](#sdxl-1024px)
    -   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Model Support

SimpleTuner supports a wide array of models, with varying levels of feature support:

*(This section has been expanded to include more details about the models and features)*

### HiDream

*   Custom ControlNet implementation (full-rank, LoRA, Lycoris).
*   Memory-efficient training for NVIDIA GPUs (AMD support planned).
*   Optional MoEGate loss augmentation.
*   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU.
*   Quantisation to reduce memory usage.

### Flux.1

*   Optimized training speed with the `--fuse_qkv_projections` option.
*   ControlNet training (full-rank, LoRA, Lycoris).
*   Instruct fine-tuning for the Kontext model.
*   Classifier-free guidance training options.
*   (optional) T5 attention masked training.
*   LyCORIS or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-torchao` for major memory savings

### Wan Video

*   Text-to-Video training.
*   LyCORIS, PEFT, and full tuning support.

### LTX Video

*   LyCORIS, PEFT, and full tuning support.

### PixArt Sigma

*   LyCORIS and full tuning support.
*   ControlNet training.
*   Two-stage PixArt training.

### NVLabs Sana

*   LyCORIS and full tuning support.

### Stable Diffusion 3

*   LoRA and full finetuning supported.
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris
*   Parameters have been optimised to get the best results, validated through from-scratch training of SD3 models

### Kwai Kolors

*   SDXL-based model with ChatGLM 6B text encoder.
*   Kolors support is almost as deep as SDXL, minus ControlNet training support.

### Lumina2

*   LoRA, Lycoris, and full finetuning are supported

### Cosmos2 Predict (Image)

*   LoRIs or full rank tuning supported.

### Legacy Stable Diffusion models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

## Hardware Requirements

*(This section has been re-organized and enhanced for clarity)*

### NVIDIA

Generally, a 3080 or higher is recommended.

### AMD

LoRA and full-rank tuning verified on 7900 XTX 24GB and MI300X. Requires more memory than Nvidia due to missing `xformers`.

### Apple

LoRA and full-rank tuning tested on M3 Max with 128GB memory, using ~12GB "Wired" and 4GB system memory for SDXL. 24GB or greater memory is likely needed for machine learning on M-series due to limitations of memory-efficient attention.

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

*   A100-80G
*   A6000-48G
*   A100-40G
*   4090-24G
*   4080-12G

### Stable Diffusion 2.x, 768px

*   16GB or better

## Toolkit

Refer to the [toolkit documentation](/toolkit/README.md) for details on the associated toolkit.

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file. Use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` for performance analysis.
For a comprehensive list of options, consult [this documentation](/OPTIONS.md).