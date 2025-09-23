# SimpleTuner: Fine-tune Diffusion Models with Ease 🚀

**SimpleTuner empowers you to easily train and fine-tune a wide range of diffusion models, from Stable Diffusion to cutting-edge architectures, with a focus on simplicity and performance.**  Discover this powerful tool on [GitHub](https://github.com/bghira/SimpleTuner)!

> **Important:**  No data is shared with third parties unless explicitly enabled through opt-in flags like `report_to`, `push_to_hub`, or manually configured webhooks.

## Key Features of SimpleTuner:

*   **Simplified Training:**  Easy-to-use with sensible defaults for a seamless experience.
*   **Versatile Compatibility:** Supports datasets of all sizes, from small projects to large-scale training.
*   **Cutting-Edge Techniques:**  Incorporates the latest, proven features for optimal results.
*   **Multi-GPU Training:** Distributed training across multiple GPUs.
*   **Advanced Caching:** Optimize training with image and caption embeddings cached to disk.
*   **Aspect Bucketing:** Supports images and videos of varied sizes and aspect ratios.
*   **Memory Optimization:** Enables training of many models on 16GB+ and 24GB GPUs with key optimizations.
*   **DeepSpeed Integration:** Train large models with gradient checkpointing and optimizer state offload.
*   **S3 Training:**  Directly train from cloud storage (Cloudflare R2, Wasabi S3).
*   **EMA Support:** Utilize Exponential Moving Average weights for enhanced stability and quality.
*   **Broad Model Support:** Compatible with various architectures including Stable Diffusion, Flux, HiDream, and more.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorials & Quickstart Guides](#tutorials--quickstart-guides)
-   [Features](#features)
    -   [Core Training Features](#core-training-features)
    -   [Model Architecture Support](#model-architecture-support)
    -   [Advanced Training Techniques](#advanced-training-techniques)
    -   [Model-Specific Features](#model-specific-features)
-   [Hardware Requirements](#hardware-requirements)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Focused on providing good default settings to minimize the need for manual configuration.
*   **Versatility:** Designed to handle diverse image quantities, from small datasets to large collections.
*   **Cutting-Edge Features:** Includes only features that have proven efficacy.

## Tutorials & Quickstart Guides

Get started quickly with our comprehensive documentation:

*   Explore the full [tutorial](/documentation/TUTORIAL.md) for in-depth understanding.
*   Jumpstart your training with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   Optimize memory usage with [DeepSpeed](/documentation/DEEPSPEED.md) for memory-constrained systems.
*   Configure multi-node distributed training with the [DISTRIBUTED guide](/documentation/DISTRIBUTED.md).
*   Quickstart guides for all supported models are also available, listed below.

## Features

SimpleTuner provides comprehensive training support across multiple diffusion model architectures with consistent feature availability:

### Core Training Features

*   Multi-GPU training
*   Advanced caching
*   Aspect bucketing
*   Memory optimization
*   DeepSpeed integration
*   S3 training
*   EMA support

### Model Architecture Support

| Model                   | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders          |
|-------------------------|------------|-----------|---------|-----------|------------|--------------|---------------|-----------------------|
| **Stable Diffusion XL** | 3.5B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L/G              |
| **Stable Diffusion 3**  | 2B-8B      | ✓         | ✓       | ✓\*       | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL     |
| **Flux.1**              | 12B        | ✓         | ✓       | ✓\*       | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL      |
| **Auraflow**            | 6.8B       | ✓         | ✓       | ✓\*       | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL              |
| **PixArt Sigma**        | 0.6B-0.9B  | ✗         | ✓       | ✓         | ✓          | int8         | ✗             | T5-XXL                |
| **Sana**                | 0.6B-4.8B  | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2-2B             |
| **Lumina2**             | 2B         | ✓         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2                |
| **Kwai Kolors**         | 5B         | ✓         | ✓       | ✓         | ✗          | ✗            | ✗             | ChatGLM-6B            |
| **LTX Video**           | 5B         | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL                |
| **Wan Video**           | 1.3B-14B   | ✓         | ✓       | ✓\*       | ✗          | int8         | ✓             | UMT5                  |
| **HiDream**             | 17B (8.5B MoE)| ✓        | ✓       | ✓\*       | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama|
| **Cosmos2**             | 2B-14B     | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | T5-XXL                |
| **OmniGen**             | 3.8B       | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL                |
| **Qwen Image**          | 20B        | ✓         | ✓       | ✓\*       | ✗          | int8/nf4 (req.)| ✓             | T5-XXL                |
| **SD 1.x/2.x (Legacy)** | 0.9B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L                |

*✓ = Supported, ✗ = Not supported, \* = Requires DeepSpeed for full-rank training*

### Advanced Training Techniques

*   TREAD (Token-wise dropout)
*   Masked loss training
*   Prior regularization
*   Gradient checkpointing
*   Loss functions (L2, Huber, Smooth L1) with scheduling support
*   SNR weighting

### Model-Specific Features

*   Flux Kontext
*   PixArt two-stage
*   Flow matching models (advanced scheduling)
*   HiDream MoE
*   T5 masked training
*   QKV fusion
*   TREAD integration
*   Classifier-free guidance

### Quickstart Guides

Detailed quickstart guides are available for all supported models:

*   [Flux.1 Guide](/documentation/quickstart/FLUX.md)
*   [Stable Diffusion 3 Guide](/documentation/quickstart/SD3.md)
*   [Stable Diffusion XL Guide](/documentation/quickstart/SDXL.md)
*   [Auraflow Guide](/documentation/quickstart/AURAFLOW.md)
*   [PixArt Sigma Guide](/documentation/quickstart/SIGMA.md)
*   [Sana Guide](/documentation/quickstart/SANA.md)
*   [Lumina2 Guide](/documentation/quickstart/LUMINA2.md)
*   [Kwai Kolors Guide](/documentation/quickstart/KOLORS.md)
*   [LTX Video Guide](/documentation/quickstart/LTXVIDEO.md)
*   [Wan Video Guide](/documentation/quickstart/WAN.md)
*   [HiDream Guide](/documentation/quickstart/HIDREAM.md)
*   [Cosmos2 Guide](/documentation/quickstart/COSMOS2IMAGE.md)
*   [OmniGen Guide](/documentation/quickstart/OMNIGEN.md)
*   [Qwen Image Guide](/documentation/quickstart/QWEN_IMAGE.md)

---

## Hardware Requirements

### General Requirements

*   **NVIDIA:** RTX 3080+ recommended (tested up to H200)
*   **AMD:** 7900 XTX 24GB and MI300X verified
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

*   **Large models (12B+)**: A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B)**: 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B)**: 12G+ sufficient for most training types

**Note**: Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific requirements.

## Setup

Install SimpleTuner easily using pip:

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

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable debug logs for detailed insights with `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`).

Analyze training loop performance using `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a complete overview of available options, refer to [this documentation](/documentation/OPTIONS.md).