# SimpleTuner: Unleash the Power of AI with Simplified Model Training

**SimpleTuner** streamlines AI model training, providing a user-friendly experience without sacrificing cutting-edge performance.  This open-source project is available on [GitHub](https://github.com/bghira/SimpleTuner).

> ℹ️  **Privacy-Focused:** SimpleTuner prioritizes your data privacy, with no data sent to third parties unless you explicitly enable reporting, hub integration, or webhooks.

## Key Features

*   **Simplified Training:** Easy-to-use with sensible default settings.
*   **Versatile:** Supports a wide range of datasets, from small to massive.
*   **Advanced Techniques:** Incorporates proven, cutting-edge training features.
*   **Multi-GPU Training:**  Leverage distributed training for faster results.
*   **Memory Optimization:** Train larger models on limited GPU memory with techniques like DeepSpeed integration, gradient checkpointing, and quantization.
*   **Model Architecture Support:** Wide range of supported models, with more being added.
*   **Cloud Storage Integration:**  Train directly from cloud storage services like S3 and R2.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial & Quickstart Guides](#tutorial-and-quickstart-guides)
*   [Features](#features)
    *   [Core Training Features](#core-training-features)
    *   [Model Architecture Support](#model-architecture-support)
    *   [Advanced Training Techniques](#advanced-training-techniques)
    *   [Model-Specific Features](#model-specific-features)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Focus on good defaults to minimize the need for extensive configuration.
*   **Versatility:** Designed to handle diverse image quantities and aspect ratios.
*   **Cutting-Edge Features:** Only proven, effective features are incorporated.

## Tutorial and Quickstart Guides

Get started quickly using the [Quick Start](/documentation/QUICKSTART.md) guide.  Comprehensive documentation is available in the [Tutorial](/documentation/TUTORIAL.md).  For memory-constrained systems, explore the [DeepSpeed document](/documentation/DEEPSPEED.md).

For multi-node distributed training, consult the [Distributed Training guide](/documentation/DISTRIBUTED.md).

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

| Model                     | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders |
|---------------------------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL**    | 3.5B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L/G      |
| **Stable Diffusion 3**     | 2B-8B      | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL |
| **Flux.1**                 | 12B        | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL |
| **Auraflow**                | 6.8B       | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL      |
| **PixArt Sigma**           | 0.6B-0.9B  | ✗         | ✓       | ✓         | ✓          | int8         | ✗             | T5-XXL        |
| **Sana**                   | 0.6B-4.8B  | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2-2B     |
| **Lumina2**                | 2B         | ✓         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2        |
| **Kwai Kolors**            | 5B         | ✓         | ✓       | ✓         | ✗          | ✗            | ✗             | ChatGLM-6B    |
| **LTX Video**              | 5B         | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL        |
| **Wan Video**              | 1.3B-14B   | ✓         | ✓       | ✓*        | ✗          | int8         | ✓             | UMT5          |
| **HiDream**                | 17B (8.5B MoE) | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama  |
| **Cosmos2**                | 2B-14B     | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | T5-XXL        |
| **OmniGen**                | 3.8B       | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL        |
| **Qwen Image**             | 20B        | ✓         | ✓       | ✓*        | ✗          | int8/nf4 (req.)| ✓             | T5-XXL        |
| **SD 1.x/2.x (Legacy)**  | 0.9B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L        |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

### Advanced Training Techniques

*   TREAD
*   Masked loss training
*   Prior regularization
*   Gradient checkpointing
*   Loss functions (L2, Huber, Smooth L1 with scheduling)
*   SNR weighting

### Model-Specific Features

*   Flux Kontext
*   PixArt two-stage
*   Flow matching models
*   HiDream MoE
*   T5 masked training
*   QKV fusion
*   TREAD integration
*   Classifier-free guidance

## Quickstart Guides

Access detailed quickstart guides for each supported model:

*   **[Flux.1 Guide](/documentation/quickstart/FLUX.md)**
*   **[Stable Diffusion 3 Guide](/documentation/quickstart/SD3.md)**
*   **[Stable Diffusion XL Guide](/documentation/quickstart/SDXL.md)**
*   **[Auraflow Guide](/documentation/quickstart/AURAFLOW.md)**
*   **[PixArt Sigma Guide](/documentation/quickstart/SIGMA.md)**
*   **[Sana Guide](/documentation/quickstart/SANA.md)**
*   **[Lumina2 Guide](/documentation/quickstart/LUMINA2.md)**
*   **[Kwai Kolors Guide](/documentation/quickstart/KOLORS.md)**
*   **[LTX Video Guide](/documentation/quickstart/LTXVIDEO.md)**
*   **[Wan Video Guide](/documentation/quickstart/WAN.md)**
*   **[HiDream Guide](/documentation/quickstart/HIDREAM.md)**
*   **[Cosmos2 Guide](/documentation/quickstart/COSMOS2IMAGE.md)**
*   **[OmniGen Guide](/documentation/quickstart/OMNIGEN.md)**
*   **[Qwen Image Guide](/documentation/quickstart/QWEN_IMAGE.md)**

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

For manual installation or development setup, consult the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable debug logs for detailed insights: `export SIMPLETUNER_LOG_LEVEL=DEBUG`.

For performance analysis, use: `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

Find a comprehensive list of options in the [OPTIONS.md documentation](/documentation/OPTIONS.md).