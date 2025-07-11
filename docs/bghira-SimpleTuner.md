# SimpleTuner: Simplify and Supercharge Your AI Model Training 🚀

**Looking to fine-tune AI models without the complexity?** SimpleTuner offers a streamlined and accessible approach, making it easy to train cutting-edge models with less fuss. [Check out the original repository!](https://github.com/bghira/SimpleTuner)

⚠️ **Warning:** *Always back up your data before running scripts, as they have the potential to damage your training data.*

SimpleTuner prioritizes simplicity and ease of use, empowering you to train a wide range of models effectively. Designed as a shared academic exercise, contributions from the community are welcome. Join our community on [Discord](https://discord.com/invite/eq3cAMZtCC) via Terminus Research Group!

## Key Features:

*   **Simplified Training:** Easy-to-use interface with good default settings to minimize tinkering.
*   **Versatile Support:** Handles various image quantities, from small datasets to extensive collections.
*   **Cutting-Edge Functionality:** Incorporates proven features, avoiding untested options.
*   **Multi-GPU Training:** Speed up training with multiple GPUs.
*   **Efficient Data Handling:** Caches images, videos, and captions to reduce memory consumption and accelerate training.
*   **Aspect Ratio Flexibility:** Supports various image and video sizes and aspect ratios, including widescreen and portrait formats.
*   **Model Compatibility:** Supports a wide range of models, including SDXL, SD3, Flux, HiDream, and more (see feature details below).
*   **DeepSpeed Integration:** Optimizes memory usage for large models, including SDXL, allowing training on lower VRAM GPUs.
*   **Low-Precision Training:** Offers options like Quantised NF4/INT8/FP8 LoRA to reduce VRAM consumption.
*   **Training from S3:** Train directly from S3-compatible storage providers, eliminating the need for local storage.
*   **ControlNet Training:** Provides support for ControlNet model training for SDXL, SD 1.x/2.x, and Flux.

## Table of Contents

*   [Key Features](#key-features)
*   [Supported Models](#supported-models)
    *   [HiDream](#hidream)
    *   [Flux.1](#flux1)
    *   [Wan Video](#wan-video)
    *   [LTX Video](#ltx-video)
    *   [PixArt Sigma](#pixart-sigma)
    *   [NVLabs Sana](#nvlabs-sana)
    *   [Stable Diffusion 3](#stable-diffusion-3)
    *   [Kwai Kolors](#kwai-kolors)
    *   [Legacy Stable Diffusion Models](#legacy-stable-diffusion-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Tutorial & Documentation](#tutorial--documentation)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Supported Models

SimpleTuner offers extensive support for various diffusion models, including:

### HiDream

*   Custom ControlNet implementation.
*   Memory-efficient training for NVIDIA GPUs.
*   Optional MoEGate loss augmentation.
*   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU.
*   Quantise the base model to `int8-quanto` or `fp8-quanto` for memory savings.

### Flux.1

*   ControlNet training via full-rank, LoRA or Lycoris.
*   Instruct fine-tuning for the Kontext \[dev] editing model.
*   Classifier-free guidance training.
*   LyCORIS or full tuning via DeepSpeed ZeRO on a single GPU.
*   Quantise the base model to `int8-quanto` or `fp8-torchao` for memory savings.

### Wan Video

*   Text to Video training is supported.
*   LyCORIS, PEFT, and full tuning all work.

### LTX Video

*   LyCORIS, PEFT, and full tuning all work.

### PixArt Sigma

*   LyCORIS and full tuning both work.
*   ControlNet training is supported.

### NVLabs Sana

*   LyCORIS and full tuning both work.

### Stable Diffusion 3

*   LoRA and full finetuning.
*   ControlNet training.

### Kwai Kolors

*   SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder.

### Legacy Stable Diffusion Models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the legacy designation.

## Hardware Requirements

### NVIDIA

*   Generally safe to use with anything 3080 and up.

### AMD

*   LoRA and full-rank tuning have been verified working on a 7900 XTX 24GB and MI300X.

### Apple

*   LoRA and full-rank tuning work on an M3 Max with 128GB of memory.  A 24G or greater machine is often recommended for machine learning with M-series hardware.

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

## Tutorial & Documentation

Before you start, explore the [tutorial](/TUTORIAL.md) for essential information. If you prefer a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide. For memory-constrained systems, consult the [DeepSpeed document](/documentation/DEEPSPEED.md).

## Toolkit

Learn more about the included [toolkit](/toolkit/README.md).

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment file (`config/config.env`). For training loop performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. See [OPTIONS.md](/OPTIONS.md) for a comprehensive list of available options.