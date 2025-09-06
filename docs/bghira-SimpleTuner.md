# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner** simplifies the process of training diffusion models, making cutting-edge features accessible for everyone.  ([See the original repository](https://github.com/bghira/SimpleTuner))

> ℹ️ No data is sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

This project prioritizes simplicity and understandability, perfect for both academic exploration and practical use. Contributions are highly encouraged!  Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for support and discussions.

## Key Features

*   **Simplicity First:** Easy to use with sensible default settings, reducing the need for complex configuration.
*   **Versatile:** Supports a wide range of image and video datasets, from small collections to massive datasets.
*   **Cutting-Edge:** Implements the latest techniques for optimal performance and model quality.
*   **Hardware-Efficient Training:** Optimized for various hardware configurations, including options for low-VRAM training.
*   **Model Support:** Extensive support for popular models, including SDXL, Stable Diffusion 3, HiDream, Flux.1, and more.
*   **Advanced Training Techniques:** Includes features like LoRA/LyCORIS, ControlNet, Mixture of Experts, Masked Loss, and prior regularization.
*   **Integration with Hugging Face Hub:** Seamless model uploads and automatic model card generation.
*   **DeepSpeed Integration:** Enables training large models with limited VRAM.
*   **Webhook Support:**  Get notified of training progress via webhooks (e.g., Discord).

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial & Quickstart](#tutorial-and-quickstart)
-   [Features](#features)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Wan Video](#wan-video)
    -   [LTX Video](#ltx-video)
    -   [PixArt Sigma](#pixart-sigma)
    -   [NVLabs Sana](#nvlabs-sana)
    -   [Stable Diffusion 3](#stable-diffusion-3)
    -   [Kwai Kolors](#kwai-kolors)
    -   [Lumina2](#lumina2)
    -   [Cosmos2 Predict (Image)](#cosmos2-predict-image)
    -   [Qwen-Image](#qwen-image)
    -   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)
-   [Hardware Requirements](#hardware-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity**: Easy setup with helpful default settings.
*   **Versatility**: Handle diverse datasets effectively.
*   **Cutting-Edge Features**: Focus on proven, effective techniques.

## Tutorial and Quickstart

For a detailed understanding, explore [the tutorial](/TUTORIAL.md).

*   **Quick Start:** For a rapid setup, use the [Quick Start guide](/documentation/QUICKSTART.md).
*   **Memory Optimization:**  Learn about DeepSpeed for memory-constrained systems in the [DeepSpeed document](/documentation/DEEPSPEED.md).
*   **Distributed Training:** Optimize for multi-node setups and billion-sample datasets with this [guide](/documentation/DISTRIBUTED.md).

## Features

*   [Multi-GPU training](#features)
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

Full training support for HiDream is included:

*   Custom ControlNet implementation for training via full-rank, LoRA or Lycoris
*   Memory-efficient training for NVIDIA GPUs (AMD support is planned)
*   Dev and Full both functioning and trainable. Fast is untested.
*   Optional MoEGate loss augmentation
*   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings
*   Quantise Llama LLM using `--text_encoder_4_precision` set to `int4-quanto` or `int8-quanto` to run on 24G cards.

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

Full training support for Flux.1 is included:

*   Double the training speed of Flux.1 with the new `--fuse_qkv_projections` option, taking advantage of Flash Attention 3 on Hopper systems
*   ControlNet training via full-rank, LoRA or Lycoris
*   Instruct fine-tuning for the Kontext \[dev] editing model implementation generously provided by [Runware](https://runware.ai).
*   Classifier-free guidance training
    *   Leave it disabled and preserve the dev model's distillation qualities
    *   Or, reintroduce CFG to the model and improve its creativity at the cost of inference speed and training time.
*   (optional) T5 attention masked training for superior fine details and generalisation capabilities
*   LoRA or full tuning via DeepSpeed ZeRO on a single GPU
*   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-torchao` for major memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

SimpleTuner has preliminary training integration for Wan 2.1 which has a 14B and 1.3B type, both of which work.

*   Text to Video training is supported.
*   Image to Video training is not yet supported.
*   Text encoder training is not supported.
*   VAE training is not supported.
*   LyCORIS, PEFT, and full tuning all work as expected
*   ControlNet training is not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

SimpleTuner has preliminary training integration for LTX Video, efficiently training on less than 16G.

*   Text encoder training is not supported
*   VAE training is not supported
*   LyCORIS, PEFT, and full tuning all work as expected
*   ControlNet training is not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.

### PixArt Sigma

SimpleTuner has extensive training integration with PixArt Sigma - both the 600M & 900M models load without modification.

*   Text encoder training is not supported
*   LyCORIS and full tuning both work as expected
*   ControlNet training is supported for full and PEFT LoRA training
*   [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support (see: [MIXTURE_OF_EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md))

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### NVLabs Sana

SimpleTuner has extensive training integration with NVLabs Sana.

This is a lightweight, fun, and fast model that makes getting into model training highly accessible to a wider audience.

*   LyCORIS and full tuning both work as expected.
*   Text encoder training is not supported.
*   PEFT Standard LoRA is not supported.
*   ControlNet training is not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide to start training.

### Stable Diffusion 3

*   LoRA and full finetuning are supported as usual.
*   ControlNet training via full-rank, PEFT LoRA, or Lycoris
*   Certain features such as segmented timestep selection and Compel long prompt weighting are not yet supported.
*   Parameters have been optimised to get the best results, validated through from-scratch training of SD3 models

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, **doubling** the hidden dimension size and substantially increasing the level of local detail included in the prompt embeds.

Kolors support is almost as deep as SDXL, minus ControlNet training support.


### Lumina2

A 2B parameter flow-matching model that uses the 16ch Flux VAE.

*   LoRA, Lycoris, and full finetuning are supported
*   ControlNet training is not yet supported

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available with example configurations.

### Cosmos2 Predict (Image)

A 2B / 14B parameter model that can do video as well as text-to-image.

*   Currently, only the text-to-image variant is supported.
*   Lycoris or full-rank tuning are supported, but PEFT LoRAs are currently not.
*   ControlNet training is not yet supported.

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available with full example configuration and dataset.

### Qwen-Image

A **massive** 20B MMDiT that can do text-to-image.

*   Lycoris, LoRA, and full-rank training are all supported, with full-rank training requiring H200 or better with DeepSpeed
*   ControlNet training is not yet supported.

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available with example configuration and dataset, as well as general training/configuration tips.

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

*   **NVIDIA:** Supports a wide range of NVIDIA GPUs.
*   **AMD:** LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.
*   **Apple:** LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory
*   **HiDream**: Details for various GPU configurations are in the [HiDream hardware requirements](#hidream) section
*   **Flux.1**:  See [Flux.1 hardware requirements](#flux1-dev-schnell) for details on various GPU configurations
*   **SDXL**: [SDXL hardware requirements](#sdxl-1024px) for different hardware setups
*   **Stable Diffusion 2.x**: Hardware requirements are outlined in the [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px) section.

## Toolkit

Explore the comprehensive [toolkit documentation](/toolkit/README.md) for more information.

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable detailed debugging by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.

Analyze the training loop's performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

Consult the [OPTIONS.md](/OPTIONS.md) for a complete list of configuration options.