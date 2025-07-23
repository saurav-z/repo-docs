# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease

SimpleTuner empowers you to fine-tune and train a wide range of diffusion models, making complex model training accessible and efficient. [[Original Repo](https://github.com/bghira/SimpleTuner)]

**Key Features:**

*   **Versatile Model Support:** Train models including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2, and Cosmos2 Predict.
*   **Hardware Optimization:** Supports training on various hardware, including NVIDIA, AMD, and Apple Silicon, with specific hardware recommendations.
*   **Memory Efficiency:** Utilizes techniques like caching, aspect bucketing, LoRA/LyCORIS, and DeepSpeed integration to reduce VRAM consumption.
*   **Advanced Training Options:** Offers features like multi-GPU training, ControlNet integration, Mixture of Experts, masked loss training, and prior regularization.
*   **Integration with Hugging Face Hub:** Seamlessly upload and share your trained models.
*   **Simplified Setup:** Provides detailed installation guides and troubleshooting tips.

**Important Considerations:**

*   No data is sent to third parties except through opt-in flags (`report_to`, `push_to_hub`) or manually configured webhooks.

**Sections in this README:**

*   [Features](#features)
*   [Supported Models](#supported-models)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Supported Models

SimpleTuner offers training support for the following diffusion models:

*   **HiDream:** Full training support, with custom ControlNet implementations and memory-efficient training options.
*   **Flux.1:** Optimized training with features like Flash Attention 3, ControlNet support, and classifier-free guidance.
*   **Wan Video:** Preliminary text-to-video training integration.
*   **LTX Video:** Preliminary training integration, with efficient memory usage.
*   **PixArt Sigma:** Extensive training integration with LyCORIS and full tuning support.
*   **NVLabs Sana:** Lightweight, fast, and accessible model training with LyCORIS and full tuning.
*   **Stable Diffusion 3:** Supports LoRA, full finetuning, and ControlNet training.
*   **Kwai Kolors:** SDXL-based model with extended hidden dimensions, optimized for detail.
*   **Lumina2:** 2B parameter flow-matching model with LoRA, Lycoris and full finetuning support.
*   **Cosmos2 Predict (Image):** Text-to-image variant support with Lycoris and full-rank tuning.

## Hardware Requirements

The following hardware requirements are recommended for optimal performance:

*   **NVIDIA:** 3080 and up is a safe bet.
*   **AMD:** Verified to work on 7900 XTX 24GB and MI300X.
*   **Apple:** Tested on M3 Max with 128G memory.

## Setup

*   Detailed setup information is available in the [installation documentation](/INSTALL.md).

## Troubleshooting

*   Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.
*   Analyze the training loop's performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   For a complete list of options, consult [this documentation](/OPTIONS.md).