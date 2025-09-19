# SimpleTuner: The Ultimate Tool for Fine-Tuning Diffusion Models 🚀

SimpleTuner simplifies the process of fine-tuning cutting-edge diffusion models, providing powerful tools for both beginners and experts.  Explore the original repository [here](https://github.com/bghira/SimpleTuner) for a deeper dive.

> ℹ️  **Data Privacy:** SimpleTuner ensures data privacy by not sending any data to third parties unless you explicitly enable `report_to`, `push_to_hub`, or manually configured webhooks.

**Key Features:**

*   **Simplified Training:** Designed for ease of use, with excellent default settings to minimize the need for configuration tweaks.
*   **Versatile Support:** Works with a wide range of image quantities and sizes.
*   **Cutting-Edge Techniques:** Implements proven features to enhance model training and performance.
*   **Multi-GPU Training:** Scale your training with distributed training across multiple GPUs.
*   **Advanced Caching:** Optimized image, video, and caption caching for faster training.
*   **Memory Optimization:** Train large models even on GPUs with limited memory using techniques like DeepSpeed.
*   **Broad Model Support:** Comprehensive support for popular architectures like Stable Diffusion XL, Stable Diffusion 3, and many more (see model support table below).

## Table of Contents

*   [Core Training Features](#core-training-features)
*   [Model Architecture Support](#model-architecture-support)
*   [Advanced Training Techniques](#advanced-training-techniques)
*   [Model-Specific Features](#model-specific-features)
*   [Quickstart Guides](#quickstart-guides)
*   [Hardware Requirements](#hardware-requirements)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Core Training Features

*   **Multi-GPU Training:** Accelerate training with distributed training across multiple GPUs.
*   **Advanced Caching:** Leverage image, video, and caption caching for faster training.
*   **Aspect Bucketing:** Train with diverse image/video sizes and aspect ratios.
*   **Memory Optimization:** Train models even on 24G GPU and many on 16G GPUs with optimization.
*   **DeepSpeed Integration:** Utilize DeepSpeed for training large models on smaller GPUs with gradient checkpointing and optimizer state offload.
*   **S3 Training:** Directly train from cloud storage (Cloudflare R2, Wasabi S3).
*   **EMA Support:** Employ Exponential Moving Average (EMA) weights for improved stability and quality.

## Model Architecture Support

SimpleTuner supports a wide range of diffusion model architectures, offering consistent feature availability across each:

| Model                    | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders     |
| ------------------------ | ---------- | --------- | ------- | --------- | ---------- | ------------ | ------------- | ----------------- |
| **Stable Diffusion XL**  | 3.5B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L/G          |
| **Stable Diffusion 3**   | 2B-8B      | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL |
| **Flux.1**               | 12B        | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL   |
| **Auraflow**             | 6.8B       | ✓         | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL          |
| **PixArt Sigma**         | 0.6B-0.9B  | ✗         | ✓       | ✓         | ✓          | int8         | ✗             | T5-XXL            |
| **Sana**                 | 0.6B-4.8B  | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2-2B         |
| **Lumina2**              | 2B         | ✓         | ✓       | ✓         | ✗          | int8         | ✓             | Gemma2            |
| **Kwai Kolors**          | 5B         | ✓         | ✓       | ✓         | ✗          | ✗            | ✗             | ChatGLM-6B        |
| **LTX Video**            | 5B         | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL            |
| **Wan Video**            | 1.3B-14B   | ✓         | ✓       | ✓*        | ✗          | int8         | ✓             | UMT5              |
| **HiDream**              | 17B (8.5B MoE) | ✓       | ✓       | ✓*        | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| **Cosmos2**              | 2B-14B     | ✗         | ✓       | ✓         | ✗          | int8         | ✓             | T5-XXL            |
| **OmniGen**              | 3.8B       | ✓         | ✓       | ✓         | ✗          | int8/fp8     | ✓             | T5-XXL            |
| **Qwen Image**           | 20B        | ✓         | ✓       | ✓*        | ✗          | int8/nf4 (req.) | ✓            | T5-XXL            |
| **SD 1.x/2.x (Legacy)** | 0.9B       | ✓         | ✓       | ✓         | ✓          | int8/nf4     | ✗             | CLIP-L            |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training

## Advanced Training Techniques

*   **TREAD:** Token-wise dropout for transformer models.
*   **Masked Loss Training:** Improved convergence with segmentation/depth guidance.
*   **Prior Regularization:** Enhance character consistency.
*   **Gradient Checkpointing:** Configurable intervals for memory/speed optimization.
*   **Loss Functions:** L2, Huber, and Smooth L1 with scheduling support.
*   **SNR Weighting:** Min-SNR gamma weighting for enhanced training dynamics.

## Model-Specific Features

*   **Flux Kontext:** Edit conditioning and image-to-image training for Flux models.
*   **PixArt two-stage:** eDiff training pipeline support for PixArt Sigma.
*   **Flow Matching Models:** Advanced scheduling with beta/uniform distributions.
*   **HiDream MoE:** Mixture of Experts gate loss augmentation.
*   **T5 masked training:** Enhanced fine details for Flux and compatible models.
*   **QKV fusion:** Memory and speed optimizations (Flux, Lumina2).
*   **TREAD integration:** Selective token routing for Wan and Flux models.
*   **Classifier-Free Guidance:** Optional CFG reintroduction for distilled models.

## Quickstart Guides

Get up and running quickly with these detailed quickstart guides for all supported models:

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

**Note:** Quantization (int8/fp8/nf4) significantly reduces memory requirements. Consult individual [quickstart guides](#quickstart-guides) for model-specific requirements.

## Setup

Install SimpleTuner using `pip`:

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

For advanced setup options, refer to the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

For detailed debug logs, set `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.  For training loop performance analysis, use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.  Consult the [options documentation](/documentation/OPTIONS.md) for comprehensive configuration details.