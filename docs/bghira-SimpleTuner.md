# SimpleTuner: Train Diffusion Models with Ease 🚀

**SimpleTuner empowers you to efficiently train state-of-the-art diffusion models with a focus on simplicity and cutting-edge techniques.**

[Explore the SimpleTuner Repository](https://github.com/bghira/SimpleTuner)

## Key Features

*   **Comprehensive Model Support:** Train a wide array of diffusion models, including Stable Diffusion XL, SD3, Flux.1, Auraflow, PixArt Sigma, and more.
*   **Multi-GPU Training:** Distribute training across multiple GPUs for faster results.
*   **Advanced Training Techniques:** Utilize features like TREAD, masked loss training, prior regularization, and SNR weighting for improved performance.
*   **Memory Optimization:** Train large models on smaller GPUs with optimizations like DeepSpeed integration, quantization, and gradient checkpointing.
*   **Flexible Data Handling:** Support for aspect bucketing, image/video caching, and training directly from cloud storage (e.g., S3).
*   **Quickstart Guides:** Get up and running quickly with detailed guides for various models.

## Table of Contents

-   [Core Training Features](#core-training-features)
-   [Model Architecture Support](#model-architecture-support)
-   [Advanced Training Techniques](#advanced-training-techniques)
-   [Model-Specific Features](#model-specific-features)
-   [Quickstart Guides](#quickstart-guides)
-   [Hardware Requirements](#hardware-requirements)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Core Training Features

*   Multi-GPU training for faster training.
*   Advanced caching for faster training with image, video, and caption embeddings cached to disk.
*   Aspect bucketing support for varying image/video sizes and aspect ratios.
*   Memory optimization and DeepSpeed integration for training on smaller GPUs.
*   S3 training support for direct training from cloud storage.
*   EMA support for improved stability and quality.

## Model Architecture Support

SimpleTuner provides consistent feature availability across multiple diffusion model architectures:

| Model                       | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders |
|-----------------------------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL**     | 3.5B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L/G      |
| **Stable Diffusion 3**      | 2B-8B      | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL |
| **Flux.1**                  | 12B        | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL |
| **Auraflow**                | 6.8B       | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL      |
| **PixArt Sigma**            | 0.6B-0.9B  | ✗         | ✓       | ✓         | ✓          | int8         | ✗             | T5-XXL        |
| **Sana**                    | 0.6B-4.8B  | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2-2B     |
| **Lumina2**                 | 2B         | ✓         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2        |
| **Kwai Kolors**             | 5B         | ✓         | ✓       | ✓         | ✗          | ✗            | ✗             | ChatGLM-6B    |
| **LTX Video**               | 5B         | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL        |
| **Wan Video**               | 1.3B-14B   | ✓         | ✓       | ✓*        | ✗          | int8         | ✓             | UMT5          |
| **HiDream**                 | 17B (8.5B MoE) | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| **Cosmos2**                 | 2B-14B     | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | T5-XXL        |
| **OmniGen**                 | 3.8B       | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL        |
| **Qwen Image**              | 20B        | ✓         | ✓       | ✓*        | ✗          | int8/nf4 (req.)| ✓             | T5-XXL        |
| **SD 1.x/2.x (Legacy)**     | 0.9B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L        |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

## Advanced Training Techniques

*   **TREAD:** Token-wise dropout for transformer models.
*   **Masked Loss Training:** Superior convergence with segmentation/depth guidance.
*   **Prior Regularization:** Enhanced training stability for character consistency.
*   **Gradient Checkpointing:** Configurable intervals for memory/speed optimization.
*   **Loss Functions:** L2, Huber, Smooth L1 with scheduling support.
*   **SNR Weighting:** Min-SNR gamma weighting for improved training dynamics.

## Model-Specific Features

*   **Flux Kontext:** Edit conditioning and image-to-image training for Flux models.
*   **PixArt two-stage:** eDiff training pipeline support for PixArt Sigma.
*   **Flow Matching Models:** Advanced scheduling with beta/uniform distributions.
*   **HiDream MoE:** Mixture of Experts gate loss augmentation.
*   **T5 Masked Training:** Enhanced fine details for Flux and compatible models.
*   **QKV Fusion:** Memory and speed optimizations (Flux, Lumina2).
*   **TREAD Integration:** Selective token routing for Wan and Flux models.
*   **Classifier-free guidance:** Optional CFG reintroduction for distilled models.

## Quickstart Guides

Detailed quickstart guides are available for all supported models, simplifying your training process.

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
*   **AMD:** 7900 XTX 24GB and MI300X verified (higher memory usage vs NVIDIA)
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

*   **Large models (12B+)**: A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B)**: 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B)**: 12G+ sufficient for most training types

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

For manual installation or development setup, refer to the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

For detailed debugging information, set the following environment variables:

*   `export SIMPLETUNER_LOG_LEVEL=DEBUG` to enable debug logs.
*   `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` to highlight training loop issues with timestamps.
*   Consult the [OPTIONS documentation](/documentation/OPTIONS.md) for available configuration.