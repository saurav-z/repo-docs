# SimpleTuner: Unleash the Power of Diffusion Model Training 🚀

**SimpleTuner** simplifies diffusion model training with a focus on ease of use, providing powerful tools for training a variety of diffusion models.  Check out the original repo: [https://github.com/bghira/SimpleTuner](https://github.com/bghira/SimpleTuner)

**Key Features:**

*   **Wide Model Support:** Train models like Stable Diffusion XL, SD3, Flux.1, and more.
*   **Advanced Training Techniques:** Utilize features like TREAD, Masked Loss, SNR weighting, and Prior Regularization.
*   **Memory Optimization:** Train large models on limited hardware with features like DeepSpeed integration, gradient checkpointing, and quantization.
*   **Multi-GPU Training:** Leverage distributed training for faster training.
*   **Quickstart Guides:** Get up and running fast with detailed guides for each supported model.
*   **Cloud Training:** Train directly from cloud storage with S3 support.
*   **Community-Focused:**  Join our community [on Discord](https://discord.gg/CVzhX7ZA).

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

*   **Multi-GPU Training:** Distribute training across multiple GPUs for faster results.
*   **Advanced Caching:** Optimize training speed with image, video, and caption caching.
*   **Aspect Bucketing:**  Efficiently handle images/videos with varied aspect ratios.
*   **Memory Optimization:** Train complex models on resource-constrained hardware.
*   **DeepSpeed Integration:** Train massive models on smaller GPUs using gradient checkpointing and optimizer state offload.
*   **S3 Training:** Train directly from cloud storage (Cloudflare R2, Wasabi S3)
*   **EMA Support:**  Improve training stability and enhance output quality using exponential moving average weights.

## Model Architecture Support

SimpleTuner supports a broad range of diffusion model architectures, offering consistent feature availability.

| Model                    | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders   |
| :----------------------- | :--------- | :-------- | :------ | :---------- | :--------- | :----------- | :------------ | :--------------- |
| **Stable Diffusion XL**  | 3.5B       | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L/G         |
| **Stable Diffusion 3**   | 2B-8B      | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL |
| **Flux.1**               | 12B        | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL   |
| **Auraflow**             | 6.8B       | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL         |
| **PixArt Sigma**         | 0.6B-0.9B  | ✗         | ✓       | ✓           | ✓          | int8         | ✗             | T5-XXL           |
| **Sana**                 | 0.6B-4.8B  | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2-2B        |
| **Lumina2**              | 2B         | ✓         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2           |
| **Kwai Kolors**          | 5B         | ✓         | ✓       | ✓           | ✗          | ✗            | ✗             | ChatGLM-6B       |
| **LTX Video**            | 5B         | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL           |
| **Wan Video**            | 1.3B-14B   | ✓         | ✓       | ✓\*         | ✗          | int8         | ✓             | UMT5             |
| **HiDream**              | 17B (8.5B) | ✓         | ✓       | ✓\*         | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| **Cosmos2**              | 2B-14B     | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | T5-XXL           |
| **OmniGen**              | 3.8B       | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL           |
| **Qwen Image**           | 20B        | ✓         | ✓       | ✓\*         | ✗          | int8/nf4     | ✓             | T5-XXL           |
| **SD 1.x/2.x (Legacy)** | 0.9B       | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L           |

*✓ = Supported, ✗ = Not supported, \* = Requires DeepSpeed for full-rank training*

## Advanced Training Techniques

*   **TREAD:** Token-wise dropout for transformer models.
*   **Masked Loss Training:**  Improved convergence with segmentation/depth guidance.
*   **Prior Regularization:** Enhanced training stability for improved character consistency.
*   **Gradient Checkpointing:** Control memory usage/training speed.
*   **Loss Functions:** L2, Huber, Smooth L1 with scheduling support.
*   **SNR Weighting:**  Min-SNR gamma weighting for enhanced training dynamics.

## Model-Specific Features

*   **Flux Kontext:** Edit conditioning and image-to-image for Flux models.
*   **PixArt two-stage:** eDiff training pipeline support for PixArt Sigma.
*   **Flow Matching Models:** Advanced scheduling with beta/uniform distributions.
*   **HiDream MoE:** Mixture of Experts gate loss augmentation.
*   **T5 Masked Training:** Enhanced detail for Flux and compatible models.
*   **QKV Fusion:**  Memory and speed optimizations (Flux, Lumina2).
*   **TREAD Integration:** Selective token routing for Wan and Flux models.
*   **Classifier-Free Guidance:** Optional CFG reintroduction for distilled models.

## Quickstart Guides

Jumpstart your training with comprehensive guides for each supported model:

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

**Note:** Quantization (int8/fp8/nf4) significantly reduces memory requirements. Refer to individual [quickstart guides](#quickstart-guides) for model-specific requirements.

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

*   **Enable Debug Logs:**  Get detailed insights by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment.
*   **Performance Analysis:** Use `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` to find configuration issues.
*   **Comprehensive Options:** Explore the full range of options in [this documentation](/documentation/OPTIONS.md).