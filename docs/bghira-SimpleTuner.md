# SimpleTuner: Simplify Your AI Model Training 🚀

**SimpleTuner** provides a user-friendly approach to fine-tuning and training various AI models, making complex processes accessible.  [Visit the GitHub repository](https://github.com/bghira/SimpleTuner) for more details.

*   **Focus on Simplicity:** Designed for ease of use with sensible default settings.
*   **Versatile:** Handles diverse image and video datasets, from small to large.
*   **Cutting-Edge Features:** Integrates effective, tested features to maximize results.
*   **Privacy-Focused:**  No data is sent to third parties except with opt-in flags or user-configured webhooks.
*   **Community:** Join the discussion on [Discord](https://discord.gg/CVzhX7ZA) through Terminus Research Group.

## Key Features

*   **Multi-GPU Training:** Boost performance with parallel processing.
*   **Advanced Dropout Techniques:** Includes [TREAD](/documentation/TREAD.md) for faster training of models like Wan 2.1/2.2 and Flux.
*   **Image/Video Caching:** Improves training speed and reduces memory usage.
*   **Aspect Bucketing:** Supports diverse image and video sizes and aspect ratios.
*   **LoRA and Full U-Net Training:** For SDXL, PixArt, SD3, SD 2.x, and other models.
*   **DeepSpeed Integration:** Allows training of SDXL's full U-Net on as little as 12GB VRAM.
*   **Quantization:** Reduces VRAM usage with NF4/INT8/FP8 LoRA training.
*   **EMA Support:** Improves model stability and generalisation.
*   **S3 Storage Support:** Train directly from S3-compatible storage providers.
*   **ControlNet Training:** Full or LoRA based for SDXL, SD 1.x/2.x, and Flux.
*   **Mixture of Experts:** Training for lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Enhances convergence and reduces overfitting.
*   **Prior Regularization Support:** For LyCORIS models.
*   **Webhook Support:** For real-time training updates.
*   **Hugging Face Hub Integration:** Seamless model upload and model card generation.

## Supported Models

*   **HiDream**
*   **Flux.1**
*   **Wan Video**
*   **LTX Video**
*   **PixArt Sigma**
*   **NVLabs Sana**
*   **Stable Diffusion 3**
*   **Kwai Kolors**
*   **Lumina2**
*   **Cosmos2 Predict**
*   **Qwen-Image**
*   **Legacy Stable Diffusion models**

## Hardware Requirements

*   **NVIDIA:** 3080 and up are generally recommended.
*   **AMD:** LoRA and full-rank tuning work on a 7900 XTX 24GB and MI300X.
*   **Apple:** M3 Max with 128GB memory (16GB+ for machine learning recommended due to memory efficiency needs).

Specific hardware requirements vary by model and are detailed in the full documentation.

## Getting Started

*   **Tutorial:** Consult the [tutorial](/TUTORIAL.md) for comprehensive instructions.
*   **Quick Start:**  For rapid setup, use the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:**  Optimize memory usage with the [DeepSpeed document](/documentation/DEEPSPEED.md).
*   **Distributed Training:** Optimize for multi-node training using the [guide](/documentation/DISTRIBUTED.md).

## Resources

*   **Toolkit:** Explore the [toolkit documentation](/toolkit/README.md).
*   **Installation:**  Find detailed setup instructions in the [installation documentation](/INSTALL.md).
*   **Troubleshooting:** Enable debug logs with `export SIMPLETUNER_LOG_LEVEL=DEBUG` and for the training loop, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` in your environment.
*   **Options:**  See the complete list of options in the [documentation](/OPTIONS.md).