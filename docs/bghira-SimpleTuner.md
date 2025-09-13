# SimpleTuner: Train Powerful AI Image Generation Models with Ease

**SimpleTuner** empowers you to fine-tune state-of-the-art AI image generation models with a focus on simplicity and ease of use. ([Original Repo](https://github.com/bghira/SimpleTuner))

> ℹ️  This project prioritizes data privacy and security; no data is sent to third parties unless explicitly enabled via flags like `report_to`, `push_to_hub`, or manually configured webhooks.

## Key Features

*   **Wide Model Support:** Train a variety of popular models, including:
    *   HiDream
    *   Flux.1
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   Qwen-Image
    *   Legacy Stable Diffusion models (SD 1.5, SD 2.x)
*   **Efficient Training:**  Optimized for speed and memory usage, supporting multi-GPU and multi-node training.
*   **Advanced Techniques:** Leverage cutting-edge features like:
    *   Custom, highly optimized image processing backend (Rust-based)
    *   New token-wise dropout techniques (TREAD)
    *   Fast aspect bucketing for versatile image/video size training
    *   LoRA/LyCORIS, full u-net training
    *   DeepSpeed integration for training on limited VRAM
    *   Quantization support for memory efficiency (NF4/INT8/FP8)
    *   Optional EMA (Exponential Moving Average) for training stability
    *   S3-compatible storage support for remote training
    *   ControlNet model training
    *   Mixture of Experts training
    *   Masked loss training for improved convergence
    *   Prior regularisation training support for LyCORIS models
    *   Webhook support
    *   Integration with Hugging Face Hub
*   **Comprehensive Documentation:**  Detailed guides and quickstart options available.

## Table of Contents

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

## Hardware Requirements

*   Detailed information on GPU compatibility is available [here](#hardware-requirements).

## Toolkit

*   Explore the [toolkit documentation](/toolkit/README.md) for more information.

## Setup

*   Get started with the [installation documentation](/INSTALL.md).

## Troubleshooting

*   Enable debug logging for detailed insights into your training process. Consult the [OPTIONS.md](/OPTIONS.md) for a complete list of available options.

---

**Join the Community:** Connect with us on [Discord](https://discord.gg/CVzhX7ZA) through Terminus Research Group for support and discussions.