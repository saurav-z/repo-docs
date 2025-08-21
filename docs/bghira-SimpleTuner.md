# SimpleTuner: Unleash Your AI Creativity with Ease 🚀

SimpleTuner empowers you to fine-tune and train a variety of cutting-edge AI models with a focus on simplicity and user-friendliness. **No data is sent to third parties unless you opt-in, keeping your data secure.**

[View the original repo on GitHub](https://github.com/bghira/SimpleTuner)

**Key Features:**

*   **Wide Model Support:** Train and fine-tune models like HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict, and Qwen-Image.
*   **Versatile Training:** Supports LoRA, LyCORIS, and full-parameter fine-tuning methods.
*   **Memory Optimization:** Features DeepSpeed integration, quantization, and other techniques to reduce VRAM consumption, enabling training on GPUs with limited memory.
*   **Advanced Techniques:** Integrates cutting-edge features such as aspect bucketing, EMA, prior regularization, and masked loss training.
*   **Hugging Face Integration:** Seamlessly integrates with the Hugging Face Hub for model uploads and dataset loading.
*   **Customizable:** Offers webhook support for monitoring, and comprehensive options for configuration.
*   **Multi-GPU Training:** Supports multi-GPU setups for faster training.
*   **Easy-to-Use:** Designed for simplicity, with good default settings and clear documentation to get you started quickly.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial & Quickstart Guides](#tutorial-and-quickstart-guides)
-   [Features](#features)
    -   [Supported Models](#supported-models)
    -   [General Training Features](#general-training-features)
-   [Hardware Requirements](#hardware-requirements)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

-   **Simplicity:** Easy-to-use with sensible defaults for common use cases.
-   **Versatility:** Supports various image/video sizes and aspect ratios.
-   **Cutting-Edge:** Only includes proven, effective features.

## Tutorial and Quickstart Guides

Explore the [tutorial](/TUTORIAL.md) for in-depth understanding.

*   [Quick Start](/documentation/QUICKSTART.md) for rapid deployment.
*   [DeepSpeed](/documentation/DEEPSPEED.md) for memory-constrained systems.
*   [Multi-Node Training](/documentation/DISTRIBUTED.md) for large datasets.

## Features

### Supported Models

*   HiDream
*   Flux.1
*   Wan Video
*   LTX Video
*   PixArt Sigma
*   NVLabs Sana
*   Stable Diffusion 3
*   Kwai Kolors
*   Lumina2
*   Cosmos2 Predict (Image)
*   Qwen-Image
*   Legacy Stable Diffusion Models

### General Training Features

*   Multi-GPU training
*   New token-wise dropout techniques like [TREAD](/documentation/TREAD.md)
*   Image, video, and caption caching for faster training
*   Aspect bucketing for diverse image sizes
*   Refiner LoRA or full u-net training for SDXL
*   LoRA/LyCORIS training for various models, using less than 16GB VRAM
*   DeepSpeed integration
*   Quantised NF4/INT8/FP8 LoRA training
*   Optional EMA (Exponential moving average)
*   Train from S3-compatible storage
*   [ControlNet model training](/documentation/CONTROLNET.md)
*   Training [Mixture of Experts](/documentation/MIXTURE_OF_EXPERTS.md)
*   [Masked loss training](/documentation/DREAMBOOTH.md#masked-loss)
*   Strong [prior regularisation](/documentation/DATALOADER.md#is_regularisation_data)
*   Webhook support
*   Integration with the [Hugging Face Hub](https://huggingface.co)

## Hardware Requirements

*   **NVIDIA:** 3080 and up recommended.
*   **AMD:** LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.
*   **Apple:** M3 Max with 128GB memory is tested and works for SDXL.
*   Specific hardware details are provided for each model in the original README.

## Toolkit

Refer to [the toolkit documentation](/toolkit/README.md) for more details.

## Setup

Detailed setup instructions can be found in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs for more insight: `export SIMPLETUNER_LOG_LEVEL=DEBUG`.

For performance analysis: `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

Consult [this documentation](/OPTIONS.md) for a complete list of options.