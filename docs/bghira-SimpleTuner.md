# SimpleTuner: Your Simplified AI Model Training Toolkit

**SimpleTuner streamlines AI model training, prioritizing ease of use and understanding.** ([See the original repository](https://github.com/bghira/SimpleTuner))

SimpleTuner is designed for simplicity, making it easy to understand and adapt. This open-source project welcomes contributions and focuses on providing accessible tools for a variety of AI training tasks.  Note that no data is sent to any third parties except through opt-in flags such as `report_to`, `push_to_hub`, or webhooks which must be manually configured.

Join our community and ask any questions on [Discord](https://discord.com/invite/eq3cAMZtCC) via Terminus Research Group.

## Key Features

*   **Versatile Model Support:** Train a wide range of diffusion models, including:
    *   Flux.1
    *   Wan 2.1 Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 2.0/2.1
    *   Stable Diffusion 3.0
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   And more...
*   **Multi-GPU Training:**  Maximize training speed and efficiency.
*   **Hardware Optimization:**  Train on GPUs with as little as 16GB of VRAM.
*   **Data Handling:**  Caches image/video data for faster, more memory-efficient training.
*   **Aspect Ratio Support:** Utilize aspect bucketing for flexible image and video sizes.
*   **LoRA & ControlNet Training:**  Fine-tune models with LoRA, LyCORIS, or full U-Net training for SDXL and other models, including ControlNet training.
*   **Advanced Techniques:** Integrate DeepSpeed, quantization, EMA, and S3-compatible storage for enhanced performance.
*   **Community Integration:**  Seamlessly upload models to the Hugging Face Hub.
*   **Webhook Support:**  Keep informed of your training progress via webhooks (e.g., Discord).

## Table of Contents

- [Tutorials & Guides](#tutorials--guides)
- [Design Philosophy](#design-philosophy)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Toolkit](#toolkit)
- [Setup](#setup)
- [Troubleshooting](#troubleshooting)

## Tutorials & Guides

*   **Tutorial:**  Start your training with our detailed [tutorial](/TUTORIAL.md).
*   **Quick Start:** Jump right in with our [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Optimize for memory-constrained systems with [DeepSpeed](/documentation/DEEPSPEED.md).
*   **Multi-Node Training:** Configure multi-node training using the guide [here](/documentation/DISTRIBUTED.md).

## Design Philosophy

*   **Simplicity:** Easy-to-use with sensible defaults.
*   **Versatility:** Handles datasets of varying sizes.
*   **Cutting-Edge:** Incorporates proven effective features.

## Hardware Requirements

Recommendations vary by model and task. Generally, NVIDIA GPUs (3080+) are well-supported. AMD, and Apple Silicon are also supported.

*   **HiDream:** A100-80G (Full tune with DeepSpeed), A100-40G (LoRA, LoKr), 3090 24G (LoRA, LoKr)
*   **Flux.1:** A100-80G (Full tune with DeepSpeed), A100-40G (LoRA, LoKr), 3090 24G (LoRA, LoKr), 4060 Ti 16G, 4070 Ti 16G, 3080 16G (int8, LoRA, LoKr), 4070 Super 12G, 3080 10G, 3060 12GB (nf4, LoRA, LoKr)
*   **SDXL:** A100-80G (EMA, large batches, LoRA @ insane batch sizes), A6000-48G (EMA@768px, no EMA@1024px, LoRA @ high batch sizes), A100-40G (EMA@1024px, EMA@768px, EMA@512px, LoRA @ high batch sizes), 4090-24G (EMA@1024px, batch size 1-4, LoRA @ medium-high batch sizes), 4080-12G (LoRA @ low-medium batch sizes)
*   **Stable Diffusion 2.x:** 16GB or better.

*(Consult the original README for more detailed hardware recommendations.)*

## Toolkit

SimpleTuner includes a useful toolkit for managing your training workflows. Refer to the [toolkit documentation](/toolkit/README.md) for details.

## Setup

Find detailed setup instructions in the [installation documentation](/INSTALL.md).

## Troubleshooting

*   **Debugging:** Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG`.
*   **Performance Analysis:** Time training loops with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   **Options:** View all available options in [this documentation](/OPTIONS.md).