# SimpleTuner: Simplify and Optimize Your AI Model Training 🚀

**SimpleTuner empowers you to train cutting-edge AI models with simplicity and efficiency, focusing on ease of use and powerful features.**  This repository offers streamlined scripts for fine-tuning a variety of models, including Stable Diffusion, HiDream, Flux, and more. **[Explore the original repository](https://github.com/bghira/SimpleTuner)**.

> ⚠️ **Important:**  Always back up your training data before using these scripts, as there's a potential for data modification during training.

**Key Features:**

*   ✅ **Simplified Training:** Designed for ease of use with sensible default settings, minimizing the need for complex configuration.
*   🖼️ **Versatile Compatibility:** Supports a wide range of image quantities and aspect ratios, from small datasets to massive collections.
*   ✨ **Cutting-Edge Technology:** Incorporates proven features like LoRA, ControlNet, and DeepSpeed for optimal performance.
*   🚄 **Efficient Training:**  Utilizes caching and aspect bucketing for faster training and reduced memory consumption.
*   🌐 **Hugging Face Integration:** Seamless model upload and automatic model card generation with the Hugging Face Hub.
*   📦 **Mixture of Experts (MoE) & Masked Loss:** Supports advanced techniques for improved model quality and convergence.
*   ☁️ **S3-Compatible Storage:** Train directly from S3-compatible storage providers like Cloudflare R2 and Wasabi S3.

## Core Capabilities

*   **Multi-GPU Training:** Leverage multiple GPUs for accelerated training.
*   **Aspect Bucketing:** Support for diverse image/video sizes and aspect ratios, enabling widescreen and portrait training.
*   **LoRA/LyCORIS & Full U-Net Training:** Efficient training for SDXL, PixArt, SD3, SD 2.x, and more, even on GPUs with limited VRAM.
*   **DeepSpeed Integration:** Allows training of large models like SDXL's full u-net on GPUs with as little as 12GB VRAM.
*   **Quantization:** Use low-precision base models to reduce VRAM usage with NF4/INT8/FP8 LoRA training.
*   **EMA (Exponential Moving Average):**  Improve training stability and reduce overfitting with optional EMA weight networks.
*   **S3-Compatible Training:** Train directly from an S3-compatible storage provider.
*   **ControlNet Training:** Full or LoRA based ControlNet model training.
*   **Mixture of Experts (MoE):** Training for lightweight, high-quality diffusion models.
*   **Masked Loss Training:** Training for superior convergence and reduced overfitting on any model
*   **Prior Regularisation:** Strong prior regularisation training support for LyCORIS models.
*   **Webhooks & Hugging Face Hub:** Webhook support for updating progress, and integration with the Hugging Face Hub for model upload.
*   **Datasets Library:** Load compatible datasets directly from the Hugging Face hub.

## Supported Models & Quickstarts

Explore training specifics and quickstart guides for each model:

*   **HiDream:** Full support, including custom ControlNet, memory-efficient training. See the [HiDream Quickstart](/documentation/quickstart/HIDREAM.md).
*   **Flux.1:** Optimized for speed with Flash Attention 3, including ControlNet and instruct fine-tuning. See the [Flux Quickstart](/documentation/quickstart/FLUX.md).
*   **Wan Video:** Text-to-Video training support. See the [Wan Video Quickstart](/documentation/quickstart/WAN.md).
*   **LTX Video:** Efficient training on less than 16G. See the [LTX Video Quickstart](/documentation/quickstart/LTXVIDEO.md).
*   **PixArt Sigma:** Extensive integration with 600M & 900M models, including ControlNet and two-stage training. See the [PixArt Quickstart](/documentation/quickstart/SIGMA.md).
*   **NVLabs Sana:**  Lightweight and fast model training. See the [NVLabs Sana Quickstart](/documentation/quickstart/SANA.md).
*   **Stable Diffusion 3:** LoRA and full finetuning are supported, as well as ControlNet training via full-rank, PEFT LoRA, or Lycoris. See the [Stable Diffusion 3 Quickstart](/documentation/quickstart/SD3.md).
*   **Kwai Kolors:** SDXL-based model with ChatGLM text encoder.
*   **Legacy Stable Diffusion:** Training support for SD 1.5 and SD 2.x models.

## Hardware Requirements

General hardware guidelines:

*   **NVIDIA:** 3080 and up recommended for general usage.
*   **AMD:** LoRA and full-rank tuning verified on 7900 XTX 24GB and MI300X.
*   **Apple:** M3 Max with 128GB or greater.
*   **Specific Models:** See the original README for model-specific hardware details.

## Resources

*   **[Installation Guide](/INSTALL.md):**  Detailed setup instructions.
*   **[Quick Start Guide](/documentation/QUICKSTART.md):**  Get started quickly.
*   **[DeepSpeed Guide](/documentation/DEEPSPEED.md):** For memory-constrained systems.
*   **[Distributed Training Guide](/documentation/DISTRIBUTED.md):**  For multi-node training.
*   **[Toolkit Documentation](/toolkit/README.md):** Learn more about the associated toolkit.
*   **[Options Documentation](/OPTIONS.md):**  Comprehensive list of configuration options.
*   **[Tutorial](/TUTORIAL.md):**  Comprehensive tutorial.
*   **[ControlNet Training](/documentation/CONTROLNET.md):** Learn about ControlNet Training.
*   **[Mixture of Experts (MoE)](/documentation/MIXTURE_OF_EXPERTS.md):** Learn about MoE Training.
*   **[Datasets Library](/documentation/data_presets/preset_subjects200k.md):** Learn about the datasets library.

## Troubleshooting

*   Enable debug logs for detailed insights by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.
*   For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.