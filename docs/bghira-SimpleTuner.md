# SimpleTuner: Train Cutting-Edge AI Models with Ease 💹

> **SimpleTuner empowers you to fine-tune a wide range of AI models with simplicity and efficiency, designed for both beginners and experts.**

[View the GitHub Repository](https://github.com/bghira/SimpleTuner)

SimpleTuner is designed for accessibility, offering a streamlined approach to AI model training. It prioritizes ease of use and understanding, making it an ideal platform for academic exploration and community contributions.  Be assured that no user data is sent to third parties by default; any integrations with external services like Hugging Face Hub, webhooks, or reporting require explicit opt-in configuration.

**Key Features:**

*   **Versatile Model Support:** Train a diverse set of models including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict, Qwen-Image, and legacy Stable Diffusion models.
*   **Hardware Optimization:** Train on a variety of GPUs (NVIDIA, AMD), with support for Apple silicon. Features include DeepSpeed integration, quantization (NF4/INT8/FP8) for reduced VRAM consumption, and S3-compatible storage support.
*   **Advanced Training Techniques:** Leverage features such as multi-GPU training, new token-wise dropout (TREAD), aspect bucketing, EMA, ControlNet support (full or LoRA), Mixture of Experts, masked loss training, and strong prior regularization.
*   **Integrated Tools:** Built-in support for Hugging Face Hub integration for seamless model sharing and management, and custom webhooks for real-time training updates.
*   **Detailed Documentation:** Comprehensive guides for setup, troubleshooting, and advanced configurations.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
    *   [HiDream](#hidream)
    *   [Flux.1](#flux-1)
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
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Designed for easy setup with sensible defaults, reducing the need for complex configurations.
*   **Versatility:** Handles a wide range of image and video datasets from small to large scales.
*   **Cutting-Edge Features:** Integrates proven and effective training techniques to achieve optimal results.

## Tutorial

For a comprehensive understanding, begin by exploring this README. For a quick start, see the [Quick Start](/documentation/QUICKSTART.md) guide. Memory-constrained systems can utilize [DeepSpeed](/documentation/DEEPSPEED.md). Explore [DISTRIBUTED.md](/documentation/DISTRIBUTED.md) for multi-node distributed training setups.

---

## Features

*   **Multi-GPU Training:** Accelerate the training process with support for multiple GPUs.
*   **TREAD Token-wise Dropout:** Fast training for Wan 2.1/2.2 and Flux, including Kontext.
*   **Caching for Faster Training:** Caches image/video/caption features to the hard drive.
*   **Aspect Bucketing:** Supports various image/video sizes and aspect ratios.
*   **Refiner and Full U-Net Training:** Supports SDXL, LoRA/LyCORIS, and  PixArt, SDXL, SD3, and SD 2.x, for models requiring less than 16G VRAM.
*   **DeepSpeed Integration:** Enables training of SDXL's full u-net with 12GB VRAM.
*   **Quantized Training:** NF4/INT8/FP8 LoRA training reduces VRAM usage.
*   **EMA Weight Network:** Optional EMA to improve training stability and prevent overfitting.
*   **S3-Compatible Storage:** Directly train from S3-compatible storage providers (e.g., Cloudflare R2, Wasabi S3).
*   **ControlNet Training:** Full or LoRA based ControlNet model training for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts:** Training of Mixture of Experts for lightweight, diffusion models.
*   **Masked Loss Training:** Improves convergence and reduces overfitting.
*   **Prior Regularisation:** Strong prior regularization support for LyCORIS models.
*   **Webhook Support:** Receive updates via webhooks to monitor training progress.
*   **Hugging Face Hub Integration:** Seamlessly upload and manage models with auto-generated model cards.
*   **Datasets Library:** Use the [datasets library](/documentation/data_presets/preset_subjects200k.md) to load compatible datasets.

### HiDream

*   **Custom ControlNet:** Full-rank, LoRA or Lycoris.
*   **Memory Efficiency:**  Optimized for NVIDIA GPUs.
*   **MoEGate Loss Augmentation (optional).**
*   **Quantisation:** Use `--base_model_precision` to `int8-quanto` or `fp8-quanto`.
*   **Llama LLM Quantisation:** Use `--text_encoder_4_precision` to `int4-quanto` or `int8-quanto`.

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   **Faster Training:**  Double the training speed with `--fuse_qkv_projections`.
*   **ControlNet Training:** Full-rank, LoRA or Lycoris.
*   **Instruct Fine-tuning:** For Kontext.
*   **Classifier-Free Guidance (optional).**
*   **T5 attention masked training (optional).**
*   **Quantisation:** Use `--base_model_precision` to `int8-quanto` or `fp8-torchao`.

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   **Text to Video Training**
*   **LyCORIS, PEFT, and full tuning**

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

*   **LyCORIS, PEFT, and full tuning**

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.

### PixArt Sigma

*   **Text encoder training is not supported**
*   **LyCORIS and full tuning**
*   **ControlNet Training**
*   **Two-stage PixArt training**

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### NVLabs Sana

*   **LyCORIS and full tuning**

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide to start training.

### Stable Diffusion 3

*   **LoRA and full finetuning are supported as usual.**
*   **ControlNet training via full-rank, PEFT LoRA, or Lycoris**

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

*   **SDXL-based model with ChatGLM**

### Lumina2

*   **LoRA, Lycoris, and full finetuning are supported**

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available.

### Cosmos2 Predict (Image)

*   **Lycoris or full-rank tuning are supported**

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available.

### Qwen-Image

*   **Lycoris, LoRA, and full-rank training are all supported**

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available.

### Legacy Stable Diffusion models

*   **RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable**

---

## Hardware Requirements

### NVIDIA

Generally, 3080 and up is a safe bet.

### AMD

Verified on a 7900 XTX 24GB and MI300X.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory.

### HiDream

*   A100-80G (Full tune with DeepSpeed)
*   A100-40G (LoRA, LoKr)
*   3090 24G (LoRA, LoKr)

### Flux.1

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

For information about the associated toolkit, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs using `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.

Use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` for performance analysis.
Consult [OPTIONS.md](/OPTIONS.md) for a comprehensive list of options.