# SimpleTuner: The Easy-to-Use Diffusion Model Trainer 🚀

**SimpleTuner** simplifies diffusion model training, making it easy to understand and use for everyone. [Explore the original repository](https://github.com/bghira/SimpleTuner).

> ⚠️ No data is sent to third parties unless you explicitly enable `report_to`, `push_to_hub`, or webhooks, which require manual configuration.

## Key Features

*   **Versatile Model Support:** Train various models including HiDream, Flux.1, Wan Video, LTX Video, PixArt Sigma, NVLabs Sana, Stable Diffusion 3, Kwai Kolors, Lumina2 and Cosmos2 Predict (Image).
*   **Hardware Flexibility:** Supports training on NVIDIA, AMD, and Apple silicon GPUs.
*   **Multi-GPU Training:** Accelerate your training with multi-GPU support.
*   **Memory Optimization:** Uses caching, aspect bucketing, and DeepSpeed integration to minimize memory consumption, enabling training on GPUs with as little as 16GB VRAM.
*   **LoRA and Full Training:** Supports both LoRA (Low-Rank Adaptation) and full U-Net training.
*   **Advanced Training Techniques:** Includes support for:
    *   Quantized NF4/INT8/FP8 LoRA training.
    *   Optional EMA (Exponential Moving Average).
    *   Training from S3-compatible storage.
    *   ControlNet model training.
    *   Mixture of Experts.
    *   Masked Loss Training.
    *   Prior Regularization.
*   **Integration:** Integrates with the Hugging Face Hub and supports webhooks for training progress updates.

## Quick Start

*   **Get started:** [Quick Start](/documentation/QUICKSTART.md)
*   **DeepSpeed for memory constrained systems:** [DeepSpeed document](/documentation/DEEPSPEED.md)
*   **Multi-node distributed training:** [this guide](/documentation/DISTRIBUTED.md)

## Hardware Requirements

*   **NVIDIA:** 3080 and up recommended.
*   **AMD:** LoRA and full-rank tuning verified on 7900 XTX 24GB and MI300X.
*   **Apple:** M3 Max with 128GB, 24G or greater recommended due to memory limitations.
*   **Model-Specific Requirements:** Refer to the documentation for specific model requirements (e.g., HiDream, Flux.1, SDXL).

## Documentation

*   [Installation documentation](/INSTALL.md).
*   [Toolkit documentation](/toolkit/README.md).
*   [Options Documentation](/OPTIONS.md)