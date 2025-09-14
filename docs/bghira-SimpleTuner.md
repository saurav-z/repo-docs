# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner empowers you to easily train and fine-tune cutting-edge diffusion models, making AI art creation accessible to everyone.**  Check out the original repo [here](https://github.com/bghira/SimpleTuner).

> ℹ️  SimpleTuner prioritizes user privacy, with no data sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or webhooks that require manual configuration.

SimpleTuner is designed for simplicity and ease of use, perfect for both beginners and experienced researchers. This toolkit provides a streamlined experience with sensible defaults, while still allowing for advanced customization. Contributions are welcome! Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for support and collaboration.

## Key Features

*   **Simplified Training:**  Focus on your creative vision, not complex setup.
*   **Wide Model Support:**  Train on a variety of diffusion models.
*   **Multi-GPU Training:**  Accelerate training with distributed computing.
*   **Memory Optimization:** Train large models on more modest hardware.
*   **Cutting-Edge Techniques:** Incorporates proven features like LoRA, Lycoris, and more.
*   **Comprehensive Documentation:** Quickstart guides and detailed tutorials to get you up and running fast.
*   **Cloud Storage Support**: Train directly from cloud storage services like S3

## Core Functionality

### Table of Contents

-   [Key Features](#key-features)
-   [Model Architecture Support](#model-architecture-support)
-   [Hardware Requirements](#hardware-requirements)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)
-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Toolkit](#toolkit)

## Design Philosophy

*   **Simplicity:** Easy to use with sensible defaults, reducing the need for extensive configuration.
*   **Versatility:** Supports a broad range of image datasets, from small to very large.
*   **Performance:** Incorporates the latest, proven training techniques to optimize results.

## Tutorial

For a quick start, begin with the [Quick Start](/documentation/QUICKSTART.md) guide.

*   Beginners should fully explore this README and the tutorial prior to training to understand key concepts.
*   For memory-constrained systems: [DeepSpeed document](/documentation/DEEPSPEED.md).
*   Multi-node distributed training: [guide](/documentation/DISTRIBUTED.md).

## Model Architecture Support

SimpleTuner provides comprehensive training support across multiple diffusion model architectures with consistent feature availability:

| Model | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders |
|-------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4 (req.) | ✓ | T5-XXL |
| **SD 1.x/2.x (Legacy)** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

### Core Training Features

*   **Multi-GPU training** - Distributed training across multiple GPUs with automatic optimization
*   **Advanced caching** - Image, video, and caption embeddings cached to disk for faster training
*   **Aspect bucketing** - Support for varied image/video sizes and aspect ratios
*   **Memory optimization** - Most models trainable on 24G GPU, many on 16G with optimizations
*   **DeepSpeed integration** - Train large models on smaller GPUs with gradient checkpointing and optimizer state offload
*   **S3 training** - Train directly from cloud storage (Cloudflare R2, Wasabi S3)
*   **EMA support** - Exponential moving average weights for improved stability and quality

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

Detailed quickstart guides are available for all supported models:

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

SimpleTuner offers straightforward installation with pip:

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

For manual installation or development setup, consult the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

For detailed debug logs, set `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (e.g., `config/config.env`).
To analyze training loop performance, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
For a comprehensive list of options, refer to the [documentation](/documentation/OPTIONS.md).

## Toolkit
The toolkit contains scripts to assist with image conversion and data preprocessing.