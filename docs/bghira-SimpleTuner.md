# SimpleTuner: Train Powerful Image Generation Models with Ease

**SimpleTuner** empowers users to train and fine-tune a diverse range of image generation models with a focus on simplicity and accessibility. ([Original Repository](https://github.com/bghira/SimpleTuner))

**Key Features:**

*   **Wide Model Support:** Compatible with a variety of models including SDXL, Stable Diffusion 3, HiDream, Flux.1, PixArt Sigma, NVLabs Sana, LTX Video, Wan Video, and more.
*   **Versatile Training Options:** Supports LoRA, LyCORIS, and full finetuning, along with ControlNet and Mixture of Experts training.
*   **Hardware Optimization:** Designed to run on a range of hardware, from 16GB GPUs to multi-GPU setups, with options for quantization and DeepSpeed integration for memory efficiency.
*   **Advanced Techniques:** Integrates cutting-edge features like TREAD dropout, aspect bucketing, masked loss training, and prior regularization for superior results.
*   **S3 and Hugging Face Integration:** Train directly from S3-compatible storage and seamlessly upload models to the Hugging Face Hub.
*   **Customization & Monitoring:** Includes webhook support for real-time progress updates and detailed debugging via environment variables.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorials & Quickstarts](#tutorials-quickstarts)
*   [Features](#features)
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
    *   [Legacy Stable Diffusion Models](#legacy-stable-diffusion-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizes ease of use with good default settings to minimize configuration.
*   **Versatility:** Supports training with datasets of all sizes.
*   **Cutting-Edge:** Incorporates proven, effective features.

## Tutorials & Quickstarts

*   **Tutorial:** Start by exploring the [main tutorial](/TUTORIAL.md) for essential information.
*   **Quick Start:** Get up and running quickly with the [Quick Start Guide](/documentation/QUICKSTART.md).
*   **DeepSpeed:** Optimize for memory-constrained systems using the [DeepSpeed document](/documentation/DEEPSPEED.md).
*   **Distributed Training:** Configure multi-node training using [this guide](/documentation/DISTRIBUTED.md).

## Features

*   Multi-GPU training
*   New token-wise dropout techniques like [TREAD](/documentation/TREAD.md) for speeding up Wan 2.1/2.2 and Flux training, including Kontext
*   Image, video, and caption features (embeds) are cached to the hard drive in advance, so that training runs faster and with less memory consumption
*   Aspect bucketing: support for a variety of image/video sizes and aspect ratios, enabling widescreen and portrait training.
*   Refiner LoRA or full u-net training for SDXL
*   Most models are trainable on a 24G GPU, or even down to 16G at lower base resolutions.
    *   LoRA/LyCORIS training for PixArt, SDXL, SD3, and SD 2.x that uses less than 16G VRAM
*   DeepSpeed integration allowing for [training SDXL's full u-net on 12G of VRAM](/documentation/DEEPSPEED.md), albeit very slowly.
*   Quantised NF4/INT8/FP8 LoRA training, using low-precision base model to reduce VRAM consumption.
*   Optional EMA (Exponential moving average) weight network to counteract model overfitting and improve training stability.
*   Train directly from an S3-compatible storage provider, eliminating the requirement for expensive local storage. (Tested with Cloudflare R2 and Wasabi S3)
*   For SDXL, SD 1.x/2.x, and Flux, full or LoRA based [ControlNet model training](/documentation/CONTROLNET.md) (not ControlLite)
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight, high-quality diffusion models
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss) for superior convergence and reduced overfitting on any model
*   Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data) training support for LyCORIS models
*   Webhook support for updating eg. Discord channels with your training progress, validations, and errors
*   Integration with the [Hugging Face Hub](https://huggingface.co) for seamless model upload and nice automatically-generated model cards.
    *   Use the [datasets library](/documentation/data_presets/preset_subjects200k.md) ([more info](/documentation/HUGGINGFACE_DATASETS.md)) to load compatible datasets directly from the hub

### HiDream

*   Custom ControlNet implementation (full-rank, LoRA, or Lycoris)
*   Memory-efficient training for NVIDIA GPUs (AMD support planned)
*   Optional MoEGate loss augmentation
*   Quantisation for memory savings

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

*   Double training speed with `--fuse_qkv_projections` (Hopper systems)
*   ControlNet training (full-rank, LoRA, or Lycoris)
*   Instruct fine-tuning for Kontext models
*   Classifier-free guidance training
*   T5 attention masked training
*   Quantisation for memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

*   Text-to-Video training support
*   LyCORIS, PEFT, and full tuning
*   ControlNet training not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md).

### LTX Video

*   LyCORIS, PEFT, and full tuning
*   ControlNet training not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md).

### PixArt Sigma

*   LyCORIS and full tuning
*   ControlNet training support
*   [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md).

### NVLabs Sana

*   LyCORIS and full tuning
*   ControlNet training not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md).

### Stable Diffusion 3

*   LoRA and full finetuning
*   ControlNet training (full-rank, PEFT LoRA, or Lycoris)

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).

### Kwai Kolors

*   SDXL-based model with ChatGLM 6B text encoder
*   No ControlNet Training

### Lumina2

*   LoRA, Lycoris, and full finetuning
*   ControlNet training not yet supported

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available.

### Cosmos2 Predict (Image)

*   Text-to-image support
*   Lycoris or full-rank tuning
*   No ControlNet support

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available.

### Qwen-Image

*   Lycoris, LoRA, and full-rank training
*   No ControlNet support

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available.

### Legacy Stable Diffusion Models

*   Supports training for RunwayML's SD 1.5 and StabilityAI's SD 2.x.

---

## Hardware Requirements

*   **NVIDIA:** 3080 and up is generally recommended.
*   **AMD:** LoRA and full-rank tuning tested on 7900 XTX and MI300X. May use more memory than equivalent NVIDIA hardware.
*   **Apple:** Tested on M3 Max with 128GB, 12G "Wired" memory + 4G system memory usage for SDXL. 24GB+ recommended.

Detailed hardware recommendations for each model are listed in the [original README](https://github.com/bghira/SimpleTuner).

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md) for information about the accompanying toolkit.

## Setup

Follow the detailed [installation documentation](/INSTALL.md) for setup instructions.

## Troubleshooting

*   Enable debug logs via `export SIMPLETUNER_LOG_LEVEL=DEBUG` to diagnose issues.
*   Analyze training loop performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   Consult [this documentation](/OPTIONS.md) for a comprehensive list of options.