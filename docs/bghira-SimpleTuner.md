# SimpleTuner: Train Powerful Diffusion Models with Ease

**SimpleTuner** simplifies diffusion model training, offering a user-friendly experience for both beginners and experts. Explore the original repository here: [SimpleTuner on GitHub](https://github.com/bghira/SimpleTuner).

Key Features:

*   **Comprehensive Model Support:** Train a wide variety of diffusion models, including Stable Diffusion, Flux.1, SD3, Auraflow, and many more.
*   **Optimized for Performance:** Features like multi-GPU training, advanced caching, and memory optimization to maximize training speed.
*   **Cutting-Edge Techniques:** Implement advanced training techniques such as TREAD, masked loss training, and SNR weighting.
*   **Hardware Flexibility:** Supports NVIDIA, AMD, and Apple Silicon GPUs with specific memory guidelines for different model sizes.
*   **Easy Setup & Installation:** Install with a simple pip command, with options for CUDA, ROCm, and Apple Silicon.
*   **Extensive Documentation:** Quickstart guides and detailed documentation for each supported model.
*   **Data Privacy Focused:** No data is sent to third parties unless you opt-in for reporting, or explicitly configure webhooks.

## Table of Contents

*   [Design Philosophy](#design-philosophy)
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

SimpleTuner is designed to be:

*   **Simple:** Offers good default settings for ease of use.
*   **Versatile:** Supports a wide range of image datasets and sizes.
*   **Cutting-Edge:** Integrates only proven and effective training techniques.

## Features

SimpleTuner provides comprehensive training support across multiple diffusion model architectures with consistent feature availability:

### Core Training Features

*   Multi-GPU training with automatic optimization
*   Advanced caching for faster training
*   Aspect bucketing for varied image/video sizes
*   Memory optimization for efficient training
*   DeepSpeed integration for large models
*   S3 training from cloud storage
*   EMA support for improved quality

### Model Architecture Support

| Model                   | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders |
| ----------------------- | ---------- | --------- | -------- | ----------- | ---------- | ------------ | ------------- | ------------- |
| **Stable Diffusion XL** | 3.5B       | ✓         | ✓        | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L/G      |
| **Stable Diffusion 3**  | 2B-8B      | ✓         | ✓        | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL |
| **Flux.1**               | 12B        | ✓         | ✓        | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL |
| **Auraflow**             | 6.8B       | ✓         | ✓        | ✓*          | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL      |
| **PixArt Sigma**        | 0.6B-0.9B  | ✗         | ✓        | ✓           | ✓          | int8         | ✗             | T5-XXL        |
| **Sana**                 | 0.6B-4.8B  | ✗         | ✓        | ✓           | ✗          | int8         | ✓             | Gemma2-2B     |
| **Lumina2**              | 2B         | ✓         | ✓        | ✓           | ✗          | int8         | ✓             | Gemma2        |
| **Kwai Kolors**          | 5B         | ✓         | ✓        | ✓           | ✗          | ✗            | ✗             | ChatGLM-6B    |
| **LTX Video**            | 5B         | ✓         | ✓        | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL        |
| **Wan Video**            | 1.3B-14B   | ✓         | ✓        | ✓*          | ✗          | int8         | ✓             | UMT5          |
| **HiDream**              | 17B (8.5B MoE) | ✓         | ✓        | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| **Cosmos2**              | 2B-14B     | ✗         | ✓        | ✓           | ✗          | int8         | ✓             | T5-XXL        |
| **OmniGen**              | 3.8B       | ✓         | ✓        | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL        |
| **Qwen Image**           | 20B        | ✓         | ✓        | ✓*          | ✗          | int8/nf4 (req.) | ✓             | T5-XXL        |
| **SD 1.x/2.x (Legacy)**   | 0.9B       | ✓         | ✓        | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L        |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

### Advanced Training Techniques

*   **TREAD:** Token-wise dropout for transformer models
*   **Masked loss training:** Improved convergence
*   **Prior regularization:** Enhanced stability
*   **Gradient checkpointing:** Memory/speed optimization
*   **Loss functions:** L2, Huber, Smooth L1 with scheduling
*   **SNR weighting:** Min-SNR gamma weighting

### Model-Specific Features

*   **Flux Kontext:** Edit conditioning and image-to-image training for Flux models
*   **PixArt two-stage:** eDiff training pipeline support
*   **Flow matching models:** Advanced scheduling
*   **HiDream MoE:** Mixture of Experts gate loss augmentation
*   **T5 masked training:** Enhanced fine details
*   **QKV fusion:** Memory and speed optimizations (Flux, Lumina2)
*   **TREAD integration:** Selective token routing (Wan and Flux)
*   **Classifier-free guidance:** CFG reintroduction for distilled models

### Quickstart Guides

Quickstart guides are available for all supported models.  Find the detailed guides at `/documentation/quickstart/MODEL_NAME.md` or by consulting the Table of Contents.

## Hardware Requirements

### General Requirements

*   **NVIDIA:** RTX 3080+ recommended (tested up to H200)
*   **AMD:** 7900 XTX 24GB and MI300X verified (higher memory usage)
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA

### Memory Guidelines by Model Size

*   **Large models (12B+):** A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B):** 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B):** 12G+ sufficient

**Note:** Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific requirements.

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

For manual installation or development, see the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable debug logs for detailed insights: `export SIMPLETUNER_LOG_LEVEL=DEBUG`.

For training loop performance analysis: `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.

Consult the documentation for available options: `/documentation/OPTIONS.md`.