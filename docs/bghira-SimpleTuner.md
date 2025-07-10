# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner simplifies the process of fine-tuning diffusion models, enabling you to create custom AI art with a user-friendly approach.**  [Explore the original repository](https://github.com/bghira/SimpleTuner) to get started!

**⚠️ Warning:**  Back up your training data! This project is designed for experimentation and may inadvertently alter your data.

## Key Features

*   **Simplified Training:** SimpleTuner streamlines the fine-tuning process for diffusion models.
*   **Versatile Model Support:** Compatible with a wide range of models, including SDXL, Stable Diffusion 3, HiDream, Flux.1, and more.
*   **Hardware Optimization:** Supports multi-GPU training, mixed-precision training, and DeepSpeed integration for efficient use of resources.
*   **Advanced Techniques:** Includes features like LoRA/LyCORIS, ControlNet training, masked loss, and prior regularization for enhanced results.
*   **Integration:** Offers integration with the Hugging Face Hub and S3-compatible storage for seamless data loading and model sharing.
*   **Extensive Documentation:**  Comprehensive guides for Setup, Troubleshooting, and specific model implementations.
*   **Active Community:** Join the community on Discord to connect, ask questions, and share your creations.

## Table of Contents

*   [Features](#features)
    *   [HiDream](#hidream)
    *   [Flux.1](#flux1)
    *   [Wan Video](#wan-video)
    *   [LTX Video](#ltx-video)
    *   [PixArt Sigma](#pixart-sigma)
    *   [NVLabs Sana](#nvlabs-sana)
    *   [Stable Diffusion 3](#stable-diffusion-3)
    *   [Kwai Kolors](#kwai-kolors)
    *   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)
*   [Tutorial](#tutorial)

## Features

*   **Multi-GPU Training:** Improve performance by training on multiple GPUs.
*   **Efficient Data Handling:** Caches images, videos, and captions to disk for faster training and reduced memory usage.
*   **Aspect Bucketing:** Supports a variety of image/video sizes and aspect ratios.
*   **LoRA and Full U-Net Training:** Fine-tune with LoRA or train the full U-Net for SDXL and other models.
*   **Low VRAM Options:** Train on GPUs with as little as 16GB of VRAM using LoRA, LyCORIS, and quantisation techniques.
*   **DeepSpeed Integration:** Utilize DeepSpeed for training SDXL on memory-constrained hardware.
*   **Quantization Support:** Training with low-precision base models (NF4/INT8/FP8 LoRA) to reduce VRAM usage.
*   **Optional EMA:** Use Exponential Moving Average to improve model stability and prevent overfitting.
*   **S3 Storage Support:** Train directly from S3-compatible storage providers (e.g., Cloudflare R2, Wasabi S3).
*   **ControlNet Training:** Train ControlNet models (full and PEFT LoRA) for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts (MoE):** Experiment with lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Superior convergence and reduced overfitting.
*   **Prior Regularization:** Enhance LyCORIS model training.
*   **Webhook Support:**  Receive updates via webhooks (e.g., Discord).
*   **Hugging Face Hub Integration:** Upload models and generate model cards easily.
*   **Hugging Face Datasets Integration:** Load compatible datasets directly from the hub.

### HiDream

*   Custom ControlNet implementation for full-rank, LoRA or Lycoris training
*   NVIDIA GPU training, AMD support planned
*   MoEGate loss augmentation (optional)
*   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise base model with `--base_model_precision` to `int8-quanto` or `fp8-quanto`

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Double training speed using `--fuse_qkv_projections`
*   ControlNet training via full-rank, LoRA or Lycoris
*   Instruct fine-tuning for the Kontext \[dev] editing model
*   Classifier-free guidance (optional)
*   T5 attention masked training
*   LoRA or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise base model using `--base_model_precision` to `int8-quanto` or `fp8-torchao`

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text to Video training supported
*   LyCORIS, PEFT, and full tuning supported
*   ControlNet training is not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide.

### LTX Video

*   LyCORIS, PEFT, and full tuning supported
*   ControlNet training is not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide.

### PixArt Sigma

*   LyCORIS and full tuning supported
*   ControlNet training supported for full and PEFT LoRA
*   Two-stage PixArt training support.

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.

### NVLabs Sana

*   LyCORIS and full tuning supported
*   ControlNet training is not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.

### Stable Diffusion 3

*   LoRA and full finetuning supported
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder.

### Legacy Stable Diffusion models

*   RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

SimpleTuner supports a wide range of hardware. Here are some general guidelines:

### NVIDIA

*   3080 and up are recommended.

### AMD

*   LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

### Apple

*   LoRA and full-rank tuning have been tested on an M3 Max with 128GB. A 24G or greater machine is recommended.

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

### SDXL, 1024px

*   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
*   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
*   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
*   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
*   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

*   16G or better

---

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md) for details on the included toolkit.

## Setup

Follow the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs to get detailed information by setting `SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`).

For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a comprehensive list of available options, consult [this documentation](/OPTIONS.md).

## Tutorial

Please read [the tutorial](/TUTORIAL.md) for key information.
For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.
For memory-constrained systems, refer to the [DeepSpeed document](/documentation/DEEPSPEED.md).
For multi-node training, see [this guide](/documentation/DISTRIBUTED.md).