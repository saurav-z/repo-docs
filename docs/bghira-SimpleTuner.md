# SimpleTuner: Simplify and Optimize Your Generative AI Model Training 🚀

**SimpleTuner** provides a streamlined and accessible platform for training and fine-tuning a variety of generative AI models, focusing on ease of use and understanding. [Learn more on GitHub](https://github.com/bghira/SimpleTuner).

## Key Features

*   **Versatile Model Support:** Train and fine-tune a wide range of models, including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, and Cosmos2 Predict.
*   **Hardware Optimization:** Supports multi-GPU training, memory-efficient techniques like DeepSpeed integration and quantization (FP8/INT8/NF4), and supports Nvidia, AMD, and Apple hardware.
*   **Data Management:** Features image, video, and caption caching, aspect bucketing, and direct training from S3-compatible storage providers.
*   **Advanced Training Options:** Offers LoRA/LyCORIS training, EMA, ControlNet integration, Mixture of Experts, masked loss training, prior regularization, and webhook support for monitoring.
*   **Seamless Integration:** Integrates with the Hugging Face Hub for model uploading and dataset loading.
*   **Simplified Setup & Debugging:** Includes detailed installation guides and debugging tools for easy setup and troubleshooting.

## Contents

*   [Features](#features)
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
    *   [Legacy Stable Diffusion Models](#legacy-stable-diffusion-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Toolkit](#toolkit)
*   [Troubleshooting](#troubleshooting)

## Model Support

SimpleTuner supports a variety of cutting-edge models. Here's a quick overview:

*   **HiDream:** Memory-efficient training for NVIDIA GPUs with custom ControlNet implementation, optional MoEGate loss augmentation, and more.  See [hardware requirements](#hidream) or the [quickstart guide](/documentation/quickstart/HIDREAM.md).
*   **Flux.1:** Faster training with new `--fuse_qkv_projections` option, ControlNet support, and Instruct fine-tuning. See [hardware requirements](#flux1-dev-schnell) or the [quickstart guide](/documentation/quickstart/FLUX.md).
*   **Wan Video:** Preliminary text-to-video training integration. See the [Wan Video Quickstart](/documentation/quickstart/WAN.md) guide to start training.
*   **LTX Video:** Efficient training with LyCORIS and full tuning. See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md) guide to start training.
*   **PixArt Sigma:** Extensive training integration with both 600M & 900M models, including ControlNet support. See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md) guide.
*   **NVLabs Sana:** Lightweight and fast model training. See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md) guide.
*   **Stable Diffusion 3:** Supports LoRA, full finetuning and ControlNet training. See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).
*   **Kwai Kolors:** An SDXL-based model with ChatGLM (General Language Model) 6B as its text encoder, almost as deep as SDXL support.
*   **Lumina2:** LoRA, Lycoris, and full finetuning support. A [Lumina2 Quickstart](/documentation/quickstart/LUMINA2.md) is available.
*   **Cosmos2 Predict (Image):** Text-to-image support with Lycoris or full-rank tuning. A [Cosmos2 Predict Quickstart](/documentation/quickstart/COSMOS2IMAGE.md) is available.
*   **Legacy Stable Diffusion Models:** Supports training with SD 1.5 and SD 2.x.

## Hardware Requirements

SimpleTuner is designed to be flexible and efficient.  See the original README for specific hardware recommendations.

## Toolkit

For information about the SimpleTuner toolkit, refer to [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment.
For performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
Consult [this documentation](/OPTIONS.md) for a comprehensive list of options.