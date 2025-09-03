# SimpleTuner: Train Powerful Diffusion Models with Ease

> **SimpleTuner empowers you to fine-tune a wide array of diffusion models, offering simplicity and cutting-edge features for optimal results.** [(Original Repo)](https://github.com/bghira/SimpleTuner)

SimpleTuner is designed for user-friendliness, making the process of training diffusion models accessible to everyone. This project is an open academic exercise, and contributions are welcome.

**Key Features:**

*   **Versatile Model Support:** Train on a wide range of models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict, and Qwen-Image.
*   **Simplified Training:** Optimized for ease of use with sensible default settings, reducing the need for extensive configuration.
*   **Multi-GPU Training:** Efficiently utilize multiple GPUs for faster training.
*   **Memory Optimization:** Features like cached image/video features, aspect bucketing, and DeepSpeed integration enable training on lower-VRAM GPUs (down to 16GB).
*   **Advanced Techniques:** Support for cutting-edge techniques like TREAD dropout, EMA, Quantised NF4/INT8/FP8 LoRA training, and Mixture of Experts training.
*   **S3 and Hugging Face Integration:** Train directly from S3-compatible storage providers and seamlessly upload models to the Hugging Face Hub.
*   **ControlNet Support:** Comprehensive ControlNet training for SDXL, SD 1.x/2.x, and Flux.
*   **Webhook Support:** Easily monitor your training progress and receive updates.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
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
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Auraflow](#auraflow)
    -   [SDXL, 1024px](#sdxl-1024px)
    -   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity:** Prioritizing easy-to-use defaults to minimize configuration.
-   **Versatility:** Designed to handle various image and video datasets.
-   **Cutting-Edge:** Includes only proven and effective features.

## Tutorial

Begin your journey with [the tutorial](/TUTORIAL.md), which contains all vital information.

For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory-constrained systems, see the [DeepSpeed document](/documentation/DEEPSPEED.md) which explains how to use 🤗Accelerate to configure Microsoft's DeepSpeed for optimiser state offload.

For multi-node distributed training, [this guide](/documentation/DISTRIBUTED.md) will help tweak the configurations from the INSTALL and Quickstart guides to be suitable for multi-node training, and optimising for image datasets numbering in the billions of samples.

## Features

*   **Multi-GPU training**
*   **Advanced Dropout Techniques:** like [TREAD](/documentation/TREAD.md) to speed up Wan 2.1/2.2 and Flux training, including Kontext.
*   **Cached Features:** Caches image, video, and caption features to the hard drive to speed up training and reduce memory usage.
*   **Aspect Bucketing:** Supports a variety of image/video sizes and aspect ratios, enabling widescreen and portrait training.
*   **Refiner and LoRA/LyCORIS Options:** Options for SDXL refiner LoRA or full u-net training
*   **Low VRAM Support:** Most models are trainable on a 24G GPU, or even down to 16G at lower base resolutions. LoRA/LyCORIS training for PixArt, SDXL, SD3, and SD 2.x use less than 16G VRAM
*   **DeepSpeed Integration:** Allowing for [training SDXL's full u-net on 12G of VRAM](/documentation/DEEPSPEED.md)
*   **Quantised Training:** Quantised NF4/INT8/FP8 LoRA training to reduce VRAM consumption.
*   **EMA:** Optional Exponential moving average weight network to counteract model overfitting and improve training stability.
*   **S3 Storage:** Train directly from an S3-compatible storage provider.
*   **ControlNet Training:** Full or LoRA based [ControlNet model training](/documentation/CONTROLNET.md) for SDXL, SD 1.x/2.x, and Flux
*   **Mixture of Experts:** Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md) for lightweight, high-quality diffusion models.
*   **Masked Loss:** [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss) for superior convergence and reduced overfitting on any model.
*   **Prior Regularisation:** Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data) training support for LyCORIS models.
*   **Webhooks:** Webhook support for updating Discord channels with training progress, validations, and errors
*   **Hugging Face Hub Integration:** Integration with the [Hugging Face Hub](https://huggingface.co) for seamless model upload and nice automatically-generated model cards.
    *   Use the [datasets library](/documentation/data_presets/preset_subjects200k.md) ([more info](/documentation/HUGGINGFACE_DATASETS.md)) to load compatible datasets directly from the hub

### HiDream

Full training support for HiDream is included:

-   Custom ControlNet implementation for training via full-rank, LoRA or Lycoris
-   Memory-efficient training for NVIDIA GPUs (AMD support is planned)
-   Dev and Full both functioning and trainable. Fast is untested.
-   Optional MoEGate loss augmentation
-   Lycoris or full tuning via DeepSpeed ZeRO on a single GPU
-   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-quanto` for major memory savings
-   Quantise Llama LLM using `--text_encoder_4_precision` set to `int4-quanto` or `int8-quanto` to run on 24G cards.

See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1

Full training support for Flux.1 is included:

-   Double the training speed of Flux.1 with the new `--fuse_qkv_projections` option, taking advantage of Flash Attention 3 on Hopper systems
-   ControlNet training via full-rank, LoRA or Lycoris
-   Instruct fine-tuning for the Kontext \[dev] editing model implementation generously provided by [Runware](https://runware.ai).
-   Classifier-free guidance training
    -   Leave it disabled and preserve the dev model's distillation qualities
    -   Or, reintroduce CFG to the model and improve its creativity at the cost of inference speed and training time.
-   (optional) T5 attention masked training for superior fine details and generalisation capabilities
-   LoRA or full tuning via DeepSpeed ZeRO on a single GPU
-   Quantise the base model using `--base_model_precision` to `int8-quanto` or `fp8-torchao` for major memory savings

See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video

SimpleTuner has preliminary training integration for Wan 2.1 which has a 14B and 1.3B type, both of which work.

-   Text to Video training is supported.
-   Image to Video training is not yet supported.
-   Text encoder training is not supported.
-   VAE training is not supported.
-   LyCORIS, PEFT, and full tuning all work as expected
-   ControlNet training is not yet supported

See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.

### LTX Video

SimpleTuner has preliminary training integration for LTX Video, efficiently training on less than 16G.

-   Text encoder training is not supported
-   VAE training is not supported
-   LyCORIS, PEFT, and full tuning all work as expected
-   ControlNet training is not yet supported

See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.

### PixArt Sigma

SimpleTuner has extensive training integration with PixArt Sigma - both the 600M & 900M models load without modification.

-   Text encoder training is not supported
-   LyCORIS and full tuning both work as expected
-   ControlNet training is supported for full and PEFT LoRA training
-   [Two-stage PixArt](https://huggingface.co/ptx0/pixart-900m-1024-ft-v0.7-stage1) training support (see: [MIXTURE_OF_EXPERTS](/documentation/MIXTURE_OF_EXPERTS.md))

See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide to start training.

### NVLabs Sana

SimpleTuner has extensive training integration with NVLabs Sana.

This is a lightweight, fun, and fast model that makes getting into model training highly accessible to a wider audience.

-   LyCORIS and full tuning both work as expected.
-   Text encoder training is not supported.
-   PEFT Standard LoRA is not supported.
-   ControlNet training is not yet supported

See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide to start training.

### Stable Diffusion 3

-   LoRA and full finetuning are supported as usual.
-   ControlNet training via full-rank, PEFT LoRA, or Lycoris
-   Certain features such as segmented timestep selection and Compel long prompt weighting are not yet supported.
-   Parameters have been optimised to get the best results, validated through from-scratch training of SD3 models

See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors

An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, **doubling** the hidden dimension size and substantially increasing the level of local detail included in the prompt embeds.

Kolors support is almost as deep as SDXL, minus ControlNet training support.

### Lumina2

A 2B parameter flow-matching model that uses the 16ch Flux VAE.

-   LoRA, Lycoris, and full finetuning are supported
-   ControlNet training is not yet supported

A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available with example configurations.

### Cosmos2 Predict (Image)

A 2B / 14B parameter model that can do video as well as text-to-image.

-   Currently, only the text-to-image variant is supported.
-   Lycoris or full-rank tuning are supported, but PEFT LoRAs are currently not.
-   ControlNet training is not yet supported.

A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available with full example configuration and dataset.

### Qwen-Image

A **massive** 20B MMDiT that can do text-to-image.

-   Lycoris, LoRA, and full-rank training are all supported, with full-rank training requiring H200 or better with DeepSpeed
-   ControlNet training is not yet supported.

A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available with example configuration and dataset, as well as general training/configuration tips.

### Legacy Stable Diffusion models

RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

---

## Hardware Requirements

### NVIDIA

Generally, 3080 and up is recommended.

### AMD

LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

Without `xformers`, more memory may be required than equivalent Nvidia hardware.

### Apple

LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory, taking about **12G** of "Wired" memory and **4G** of system memory for SDXL.

  *   You likely need a 24G or greater machine for machine learning with M-series hardware due to the lack of memory-efficient attention.
  *   Subscribing to Pytorch issues for MPS is probably a good idea, as random bugs will make training stop working.

### HiDream [dev, full]

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)

HiDream has not been tested on 16G cards, but with aggressive quantisation and pre-caching of embeds, you might make it work, though even 24G is pushing limits.

### Flux.1 [dev, schnell]

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)
-   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
-   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

Flux prefers being trained with multiple large GPUs but a single 16G card should be able to do it with quantisation of the transformer and text encoders.

Kontext requires a bit beefier compute and memory allocation; a 4090 will go from ~3 to ~6 seconds per step when it is enabled.

### Auraflow

-   A100-80G (Full tune with DeepSpeed)
-   A100-40G (LoRA, LoKr)
-   3090 24G (LoRA, LoKr)
-   4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
-   4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px

-   A100-80G (EMA, large batches, LoRA @ insane batch sizes)
-   A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
-   A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
-   4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
-   4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px

-   16G or better

## Toolkit

For more information about the associated toolkit distributed with SimpleTuner, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/OPTIONS.md).