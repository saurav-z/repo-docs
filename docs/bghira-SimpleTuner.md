# SimpleTuner: Effortlessly Tune Diffusion Models for Cutting-Edge AI Generation

**SimpleTuner** empowers you to fine-tune diffusion models with remarkable ease and flexibility, offering robust features and streamlined workflows. ([View on GitHub](https://github.com/bghira/SimpleTuner))

> ℹ️  SimpleTuner prioritizes your data privacy; no data is sent to third parties unless explicitly enabled via `report_to`, `push_to_hub`, or custom webhooks.

**Key Features:**

*   **Simplified Training:** Designed for ease of use, offering sensible defaults and minimal configuration.
*   **Broad Model Support:** Train a wide variety of diffusion models, including Stable Diffusion, Flux.1, Auraflow, PixArt Sigma, and many more.
*   **Multi-GPU & Optimization:** Leverage multi-GPU training, caching, and memory optimization for efficient training.
*   **Advanced Techniques:** Implement cutting-edge techniques like TREAD, Masked Loss Training, and SNR weighting.
*   **Comprehensive Hardware Support:** Compatible with NVIDIA, AMD, and Apple Silicon GPUs.
*   **DeepSpeed Integration:** Train large models on smaller GPUs with gradient checkpointing and optimizer state offload.
*   **Quickstart Guides:** Detailed guides for each supported model to get you up and running quickly.

## Table of Contents

*   [Features](#features)
    *   [Core Training Features](#core-training-features)
    *   [Model Architecture Support](#model-architecture-support)
    *   [Advanced Training Techniques](#advanced-training-techniques)
    *   [Model-Specific Features](#model-specific-features)
    *   [Quickstart Guides](#quickstart-guides)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

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

| Model                   | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders         |
| ----------------------- | ---------- | --------- | ------- | ----------- | ---------- | ------------ | ------------- | --------------------- |
| **Stable Diffusion XL** | 3.5B       | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L/G              |
| **Stable Diffusion 3**  | 2B-8B      | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL     |
| **Flux.1**              | 12B        | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL       |
| **Auraflow**            | 6.8B       | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL              |
| **PixArt Sigma**        | 0.6B-0.9B  | ✗         | ✓       | ✓           | ✓          | int8         | ✗             | T5-XXL                |
| **Sana**                | 0.6B-4.8B  | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2-2B             |
| **Lumina2**             | 2B         | ✓         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2                |
| **Kwai Kolors**         | 5B         | ✓         | ✓       | ✓           | ✗          | ✗            | ✗             | ChatGLM-6B            |
| **LTX Video**           | 5B         | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL                |
| **Wan Video**           | 1.3B-14B   | ✓         | ✓       | ✓\*         | ✗          | int8         | ✓             | UMT5                  |
| **HiDream**             | 17B (8.5B MoE) | ✓   | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| **Cosmos2**             | 2B-14B     | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | T5-XXL                |
| **OmniGen**             | 3.8B       | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL                |
| **Qwen Image**          | 20B        | ✓         | ✓       | ✓\*         | ✗          | int8/nf4 (req.) | ✓             | T5-XXL                |
| **SD 1.x/2.x (Legacy)** | 0.9B       | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L                |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training

### Advanced Training Techniques

*   TREAD
*   Masked loss training
*   Prior regularization
*   Gradient checkpointing
*   Loss functions
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

---

## Hardware Requirements

### General Requirements

*   **NVIDIA**: RTX 3080+ recommended (tested up to H200)
*   **AMD**: 7900 XTX 24GB and MI300X verified (higher memory usage vs NVIDIA)
*   **Apple**: M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

*   **Large models (12B+)**: A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B)**: 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B)**: 12G+ sufficient for most training types

**Note**: Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific requirements.

## Setup

SimpleTuner can be installed via pip:

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

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG`.

For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

For a comprehensive list of options, consult [this documentation](/documentation/OPTIONS.md).