# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease 🚀

**SimpleTuner empowers you to train and fine-tune state-of-the-art diffusion models with simplicity and speed.** (View the original repo [here](https://github.com/bghira/SimpleTuner).)

> ℹ️  SimpleTuner prioritizes your data privacy. No data is sent to third parties unless you explicitly enable the `report_to` flag, `push_to_hub`, or webhooks, which require manual configuration.

Built with a focus on user-friendliness, SimpleTuner is an excellent tool for both academic research and practical applications.  The project welcomes contributions from the community! Join our community and ask questions on [Discord](https://discord.gg/JGkSwEbjRb) via Terminus Research Group.

## Key Features

*   **Multi-GPU Training:** Distribute training across multiple GPUs for faster results.
*   **Advanced Caching:** Accelerate training with optimized image, video, and caption caching.
*   **Aspect Bucketing:** Train on diverse image and video sizes and aspect ratios.
*   **Memory Optimization:** Train on a range of GPUs, even those with limited memory (e.g., 16GB, 24GB).
*   **DeepSpeed Integration:** Utilize DeepSpeed for large models and smaller GPUs, supporting gradient checkpointing and optimizer state offload.
*   **Cloud Storage Support:** Train directly from cloud storage services like Cloudflare R2 and Wasabi S3.
*   **EMA Support:** Enhance training stability and improve image quality with Exponential Moving Average (EMA) weights.
*   **Extensive Model Support:** Compatible with a wide array of diffusion model architectures including Stable Diffusion XL/3, Flux.1, Auraflow, PixArt Sigma, Sana, Lumina2, Kwai Kolors, LTX Video, Wan Video, HiDream, Cosmos2, OmniGen, Qwen Image, and SD 1.x/2.x (Legacy).
*   **Advanced Training Techniques:** Utilize features like TREAD, Masked Loss Training, Prior Regularization, Gradient Checkpointing, various Loss Functions and SNR weighting for superior training dynamics.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
    *   [Core Training Features](#core-training-features)
    *   [Model Architecture Support](#model-architecture-support)
    *   [Advanced Training Techniques](#advanced-training-techniques)
    *   [Model-Specific Features](#model-specific-features)
    *   [Quickstart Guides](#quickstart-guides)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:**  Default settings are designed for optimal performance, minimizing the need for complex configuration.
*   **Versatility:** Handles datasets of varying sizes, from small to extremely large collections.
*   **Cutting-Edge:** Focuses on proven, effective features, avoiding untested options.

## Tutorial

Begin your SimpleTuner journey by exploring this README. Then, delve into the [tutorial](/documentation/TUTORIAL.md) for a comprehensive guide.

For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.

For memory-constrained systems, explore the [DeepSpeed document](/documentation/DEEPSPEED.md) on utilizing 🤗Accelerate and Microsoft's DeepSpeed for optimizer state offload.

For distributed training across multiple nodes, see the [guide](/documentation/DISTRIBUTED.md) to optimize configurations for multi-node training and large image datasets.

## Features

SimpleTuner provides comprehensive training support for a range of diffusion model architectures, with consistent feature availability:

### Core Training Features

*   **Multi-GPU training** - Distributed training across multiple GPUs with automatic optimization
*   **Advanced caching** - Image, video, and caption embeddings cached to disk for faster training
*   **Aspect bucketing** - Support for varied image/video sizes and aspect ratios
*   **Memory optimization** - Most models trainable on 24G GPU, many on 16G with optimizations
*   **DeepSpeed integration** - Train large models on smaller GPUs with gradient checkpointing and optimizer state offload
*   **S3 training** - Train directly from cloud storage (Cloudflare R2, Wasabi S3)
*   **EMA support** - Exponential moving average weights for improved stability and quality

### Model Architecture Support

| Model                 | Parameters  | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders       |
| --------------------- | ----------- | --------- | ------- | ----------- | ---------- | ------------ | ------------- | ------------------- |
| **Stable Diffusion XL** | 3.5B        | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L/G            |
| **Stable Diffusion 3**  | 2B-8B       | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL   |
| **Flux.1**              | 12B         | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL     |
| **Auraflow**            | 6.8B        | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL            |
| **PixArt Sigma**        | 0.6B-0.9B   | ✗         | ✓       | ✓           | ✓          | int8         | ✗             | T5-XXL              |
| **Sana**                | 0.6B-4.8B   | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2-2B           |
| **Lumina2**             | 2B          | ✓         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2              |
| **Kwai Kolors**         | 5B          | ✓         | ✓       | ✓           | ✗          | ✗            | ✗             | ChatGLM-6B          |
| **LTX Video**           | 5B          | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL              |
| **Wan Video**           | 1.3B-14B    | ✓         | ✓       | ✓*          | ✗          | int8         | ✓             | UMT5                |
| **HiDream**             | 17B (8.5B MoE) | ✓      | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| **Cosmos2**             | 2B-14B      | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | T5-XXL              |
| **OmniGen**             | 3.8B        | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL              |
| **Qwen Image**          | 20B         | ✓         | ✓       | ✓*          | ✗          | int8/nf4 (req.) | ✓             | T5-XXL              |
| **SD 1.x/2.x (Legacy)** | 0.9B        | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L              |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

### Advanced Training Techniques

*   **TREAD** - Token-wise dropout for transformer models, including Kontext training
*   **Masked loss training** - Superior convergence with segmentation/depth guidance
*   **Prior regularization** - Enhanced training stability for character consistency
*   **Gradient checkpointing** - Configurable intervals for memory/speed optimization
*   **Loss functions** - L2, Huber, Smooth L1 with scheduling support
*   **SNR weighting** - Min-SNR gamma weighting for improved training dynamics

### Model-Specific Features

*   **Flux Kontext** - Edit conditioning and image-to-image training for Flux models
*   **PixArt two-stage** - eDiff training pipeline support for PixArt Sigma
*   **Flow matching models** - Advanced scheduling with beta/uniform distributions
*   **HiDream MoE** - Mixture of Experts gate loss augmentation
*   **T5 masked training** - Enhanced fine details for Flux and compatible models
*   **QKV fusion** - Memory and speed optimizations (Flux, Lumina2)
*   **TREAD integration** - Selective token routing for Wan and Flux models
*   **Classifier-free guidance** - Optional CFG reintroduction for distilled models

### Quickstart Guides

Find detailed quickstart guides for all supported models here:

*   **[Flux.1 Guide](/documentation/quickstart/FLUX.md)** - Includes Kontext editing support and QKV fusion
*   **[Stable Diffusion 3 Guide](/documentation/quickstart/SD3.md)** - Full and LoRA training with ControlNet
*   **[Stable Diffusion XL Guide](/documentation/quickstart/SDXL.md)** - Complete SDXL training pipeline
*   **[Auraflow Guide](/documentation/quickstart/AURAFLOW.md)** - Flow-matching model training
*   **[PixArt Sigma Guide](/documentation/quickstart/SIGMA.md)** - DiT model with two-stage support
*   **[Sana Guide](/documentation/quickstart/SANA.md)** - Lightweight flow-matching model
*   **[Lumina2 Guide](/documentation/quickstart/LUMINA2.md)** - 2B parameter flow-matching model
*   **[Kwai Kolors Guide](/documentation/quickstart/KOLORS.md)** - SDXL-based with ChatGLM encoder
*   **[LTX Video Guide](/documentation/quickstart/LTXVIDEO.md)** - Video diffusion training
*   **[Wan Video Guide](/documentation/quickstart/WAN.md)** - Video flow-matching with TREAD support
*   **[HiDream Guide](/documentation/quickstart/HIDREAM.md)** - MoE model with advanced features
*   **[Cosmos2 Guide](/documentation/quickstart/COSMOS2IMAGE.md)** - Multi-modal image generation
*   **[OmniGen Guide](/documentation/quickstart/OMNIGEN.md)** - Unified image generation model
*   **[Qwen Image Guide](/documentation/quickstart/QWEN_IMAGE.md)** - 20B parameter large-scale training

---

## Hardware Requirements

### General Requirements

*   **NVIDIA:** RTX 3080+ recommended (tested up to H200)
*   **AMD:** 7900 XTX 24GB and MI300X verified (higher memory usage vs NVIDIA)
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

*   **Large models (12B+)**: A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B)**: 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B)**: 12G+ sufficient for most training types

**Note**: Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific requirements.

## Setup

Install SimpleTuner using pip:

```bash
# Base installation (CPU-only PyTorch)
pip install simpletuner

# CUDA users (NVIDIA GPUs)
pip install simpletuner[cuda]

# ROCm users (AMD GPUs)
pip install simpletuner[rocm]

# Apple Silicon users (M1/M2/M3/M4 Macs)
pip install simpletuner[apple]
```

For manual or development installations, see the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable detailed debugging by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`) file.

Analyze the training loop performance by setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` for timestamp-based issue identification.

Consult [this documentation](/documentation/OPTIONS.md) for a complete list of available options.