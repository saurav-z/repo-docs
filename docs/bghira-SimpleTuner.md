# SimpleTuner: Train State-of-the-Art Diffusion Models with Ease 🚀

SimpleTuner simplifies diffusion model training, offering a user-friendly approach to achieve impressive results. Built with a focus on clarity and efficiency, SimpleTuner empowers you to fine-tune models with ease.

[View the original repository](https://github.com/bghira/SimpleTuner)

**Key Features:**

*   **Versatile Model Support:** Train a wide range of models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, and more!
*   **Memory Optimization:** Employ advanced techniques like aspect bucketing, caching, and DeepSpeed integration to handle diverse datasets and reduce VRAM consumption, enabling training on GPUs with as little as 16GB of VRAM.
*   **Cutting-Edge Techniques:** Leverage features like TREAD dropout, EMA, quantisation, and ControlNet training for superior results and model stability.
*   **Simplified Configuration:** Utilize sensible default settings, and access comprehensive documentation for setup and training, minimizing the need for extensive customization.
*   **Integration with Hugging Face Hub:** Seamlessly upload and manage your trained models with automatic model card generation.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Key Features](#key-features)
*   [Model Support](#model-support)
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

*   **Simplicity:** Good default settings to minimize the need for advanced configuration.
*   **Versatility:** Supports a wide array of image/video quantities and aspect ratios.
*   **Cutting-Edge:** Incorporates features proven to improve training and model quality.

## Model Support

SimpleTuner offers extensive integration for various diffusion models:

### HiDream
- Custom ControlNet, memory-efficient training, MoEGate loss augmentation, and more.
- See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).

### Flux.1
- Optimized with double the training speed with `--fuse_qkv_projections`.
- See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).

### Wan Video
- Preliminary training integration for Wan 2.1 with LyCORIS, PEFT, and full tuning.
- See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide.

### LTX Video
- Preliminary training integration for LTX Video with LyCORIS, PEFT, and full tuning, efficient on < 16G.
- See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide.

### PixArt Sigma
- Extensive integration with PixArt Sigma, including full and PEFT LoRA ControlNet training.
- See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.

### NVLabs Sana
- Extensive training integration with NVLabs Sana for LyCORIS and full tuning.
- See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.

### Stable Diffusion 3
- Full support for training and fine-tuning SD3 models, offering LoRA and ControlNet implementation.
- See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md) to get going.

### Kwai Kolors
- An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, **doubling** the hidden dimension size and substantially increasing the level of local detail included in the prompt embeds.

### Lumina2
- A 2B parameter flow-matching model that uses the 16ch Flux VAE.
- A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available with example configurations.

### Cosmos2 Predict (Image)
- A 2B / 14B parameter model that can do video as well as text-to-image.
- A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available with full example configuration and dataset.

### Qwen-Image
- A **massive** 20B MMDiT that can do text-to-image.
- A [Qwen Image Quickstart](/documentation/quickstart/QWEN_IMAGE.md) is available with example configuration and dataset, as well as general training/configuration tips.

### Legacy Stable Diffusion Models
- RunwayML's SD 1.5 and StabilityAI's SD 2.x are both trainable under the `legacy` designation.

## Hardware Requirements

*   **NVIDIA:** 3080 and up is a safe bet. YMMV.
*   **AMD:** LoRA and full-rank tuning verified working on a 7900 XTX 24GB and MI300X.
*   **Apple:** Tested to work on an M3 Max with 128GB.  Requires 24GB for machine learning.

### HiDream [dev, full]
- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)

### Flux.1 [dev, schnell]
- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### Auraflow
- A100-80G (Full tune with DeepSpeed)
- A100-40G (LoRA, LoKr)
- 3090 24G (LoRA, LoKr)
- 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr)
- 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)

### SDXL, 1024px
- A100-80G (EMA, large batches, LoRA @ insane batch sizes)
- A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes)
- A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes)
- 4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes)
- 4080-12G (LoRA @ low-medium batch sizes)

### Stable Diffusion 2.x, 768px
- 16G or better

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md) for more information about the tools included.

## Setup

Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enhance debugging by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment for detailed logs, and `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` for performance insights. For more options, consult [this documentation](/OPTIONS.md).