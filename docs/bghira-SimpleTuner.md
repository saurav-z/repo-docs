# SimpleTuner: Your Gateway to Simplified AI Model Training 💹

**SimpleTuner empowers you to train cutting-edge AI models with ease, focusing on simplicity and versatility.** ([View on GitHub](https://github.com/bghira/SimpleTuner))

> ℹ️ No data is sent to third parties unless you opt-in via `report_to`, `push_to_hub`, or manually configured webhooks.

SimpleTuner is designed for straightforward AI model training, making it accessible for both academic and enthusiast users. Contributions are welcome to help improve this shared project. Join the community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for support and discussion.

## Key Features

*   **Simplified Training:** Focuses on ease of use with sensible defaults, reducing the need for complex configurations.
*   **Versatile Compatibility:** Supports a wide array of image and video datasets, accommodating various sizes and aspect ratios.
*   **Cutting-Edge Model Support:** Train models like HiDream, Flux.1, Wan 2.1 Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, Cosmos2 Predict, and Qwen-Image.
*   **Multi-GPU Training:** Leverages multi-GPU setups for faster training.
*   **Advanced Techniques:** Implements features like TREAD dropout, aspect bucketing, and masked loss training.
*   **DeepSpeed Integration:** Includes DeepSpeed for memory optimization, enabling training on lower-VRAM GPUs.
*   **Quantization:** Supports NF4/INT8/FP8 LoRA training for reduced VRAM usage.
*   **S3-Compatible Storage:** Trains directly from S3-compatible storage providers.
*   **Hugging Face Hub Integration:** Offers seamless model upload and model card generation.
*   **Webhook Support:** Provides webhooks for real-time training progress updates.
*   **Comprehensive Documentation:** Detailed guides and quickstarts for various models.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorials and Quickstarts](#tutorials-and-quickstarts)
*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizes good default settings for ease of use.
*   **Versatility:** Designed to handle a wide range of image quantities and aspect ratios.
*   **Cutting-Edge Features:** Incorporates proven features for optimal results.

## Tutorials and Quickstarts

*   For detailed information, begin with the main [Tutorial](/TUTORIAL.md).
*   For a quick start, consult the [Quick Start](/documentation/QUICKSTART.md) guide.
*   Optimize training on memory-constrained systems with the [DeepSpeed document](/documentation/DEEPSPEED.md).
*   For multi-node distributed training, review the [guide](/documentation/DISTRIBUTED.md).

## Supported Models

Detailed quickstart guides are available for each of the following models:

*   [HiDream](/documentation/quickstart/HIDREAM.md)
*   [Flux.1](/documentation/quickstart/FLUX.md)
*   [Wan Video](/documentation/quickstart/WAN.md)
*   [LTX Video](/documentation/quickstart/LTXVIDEO.md)
*   [PixArt Sigma](/documentation/quickstart/SIGMA.md)
*   [NVLabs Sana](/documentation/quickstart/SANA.md)
*   [Stable Diffusion 3](/documentation/quickstart/SD3.md)
*   [Kwai Kolors](#kwai-kolors)
*   [Lumina2](/documentation/quickstart/LUMINA2.md)
*   [Cosmos2 Predict (Image)](/documentation/quickstart/COSMOS2IMAGE.md)
*   [Qwen-Image](/documentation/quickstart/QWEN_IMAGE.md)
*   [Legacy Stable Diffusion models](#legacy-stable-diffusion-models)

## Hardware Requirements

### NVIDIA
*   Any NVIDIA GPU with a 3080 or higher (YMMV).

### AMD
*   LoRA and full-rank tuning verified on 7900 XTX 24GB and MI300X.

### Apple
*   LoRA and full-rank tuning tested on M3 Max with 128GB memory.

### Specific Model Requirements:

*   **HiDream:** A100-80G (Full tune with DeepSpeed), A100-40G (LoRA, LoKr), 3090 24G (LoRA, LoKr).
*   **Flux.1:** A100-80G (Full tune with DeepSpeed), A100-40G (LoRA, LoKr), 3090 24G (LoRA, LoKr), 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr), 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr).
*   **Auraflow:** A100-80G (Full tune with DeepSpeed), A100-40G (LoRA, LoKr), 3090 24G (LoRA, LoKr), 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr), 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr).
*   **SDXL, 1024px:** A100-80G, A6000-48G, A100-40G, 4090-24G, 4080-12G
*   **Stable Diffusion 2.x, 768px:** 16G or better.

## Toolkit

Explore the SimpleTuner toolkit documentation for more details: [the toolkit documentation](/toolkit/README.md).

## Setup

Refer to the [installation documentation](/INSTALL.md) for detailed setup instructions.

## Troubleshooting

*   Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.
*   Analyze training loop performance using `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   For a comprehensive list of options, consult [this documentation](/OPTIONS.md).