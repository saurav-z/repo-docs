# SimpleTuner: Unleash the Power of Diffusion Models with Ease

**SimpleTuner** is a powerful and user-friendly toolkit designed for fine-tuning a wide range of diffusion models, making complex training accessible to everyone. ([Original Repo](https://github.com/bghira/SimpleTuner))

> **Privacy First:** SimpleTuner prioritizes user privacy by default, with no data sent to third parties unless explicitly enabled via flags like `report_to`, `push_to_hub`, or manually configured webhooks.

## Key Features:

*   **Versatile Model Support:** Train a diverse array of diffusion models including Stable Diffusion XL, SD3, Flux.1, Auraflow, PixArt Sigma, and many more.
*   **Advanced Training Techniques:** Leverage cutting-edge features like TREAD, Masked Loss Training, Prior Regularization, Gradient Checkpointing, and SNR weighting for optimized training.
*   **Memory Optimization:** Train large models on smaller GPUs with optimizations like DeepSpeed integration, quantization, and S3 training, reducing memory requirements.
*   **Multi-GPU & Cloud Training:** Take advantage of Multi-GPU training, aspect bucketing, advanced caching, and EMA support.
*   **Quickstart Guides:** Get up and running quickly with detailed quickstart guides for all supported models.

## Key Features Breakdown:

### Core Training Features

*   **Multi-GPU Training:** Distributed training across multiple GPUs.
*   **Advanced Caching:** Image, video, and caption embeddings cached to disk.
*   **Aspect Bucketing:** Support for varied image/video sizes and aspect ratios.
*   **Memory Optimization:** Train on 16GB/24GB GPUs using DeepSpeed and other techniques.
*   **DeepSpeed Integration:** Leverage gradient checkpointing and optimizer state offload.
*   **S3 Training:** Train directly from cloud storage (Cloudflare R2, Wasabi S3).
*   **EMA Support:** Exponential moving average weights for improved stability.

### Model Architecture Support

| Model                  | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders          |
| ---------------------- | :-------- | :------ | :-------- | :---------- | :----------- | :------------ | :--------------------- |
| Stable Diffusion XL    | ✓         | ✓       | ✓         | ✓           | int8/nf4     | ✗             | CLIP-L/G               |
| Stable Diffusion 3     | ✓         | ✓       | ✓*        | ✓           | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL      |
| Flux.1                 | ✓         | ✓       | ✓*        | ✓           | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL        |
| Auraflow               | ✓         | ✓       | ✓*        | ✓           | int8/fp8/nf4 | ✓             | UMT5-XXL               |
| PixArt Sigma           | ✗         | ✓       | ✓         | ✓           | int8         | ✗             | T5-XXL                 |
| Sana                   | ✗         | ✓       | ✓         | ✗           | int8         | ✓             | Gemma2-2B              |
| Lumina2                | ✓         | ✓       | ✓         | ✗           | int8         | ✓             | Gemma2                 |
| Kwai Kolors            | ✓         | ✓       | ✓         | ✗           | ✗            | ✗             | ChatGLM-6B             |
| LTX Video              | ✓         | ✓       | ✓         | ✗           | int8/fp8     | ✓             | T5-XXL                 |
| Wan Video              | ✓         | ✓       | ✓*        | ✗           | int8         | ✓             | UMT5                   |
| HiDream                | ✓         | ✓       | ✓*        | ✓           | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| Cosmos2                | ✗         | ✓       | ✓         | ✗           | int8         | ✓             | T5-XXL                 |
| OmniGen                | ✓         | ✓       | ✓         | ✗           | int8/fp8     | ✓             | T5-XXL                 |
| Qwen Image             | ✓         | ✓       | ✓*        | ✗           | int8/nf4     | ✓             | T5-XXL                 |
| SD 1.x/2.x (Legacy) | ✓         | ✓       | ✓         | ✓           | int8/nf4     | ✗             | CLIP-L                 |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

### Advanced Training Techniques

*   **TREAD:** Token-wise dropout for transformer models.
*   **Masked Loss Training:** Superior convergence with segmentation/depth guidance.
*   **Prior Regularization:** Enhanced training stability.
*   **Gradient Checkpointing:** Configurable intervals for memory/speed optimization.
*   **Loss Functions:** L2, Huber, Smooth L1 with scheduling support.
*   **SNR Weighting:** Min-SNR gamma weighting for improved training dynamics.

### Model-Specific Features

*   **Flux Kontext:** Edit conditioning and image-to-image training for Flux models.
*   **PixArt Two-Stage:** eDiff training pipeline support for PixArt Sigma.
*   **Flow Matching Models:** Advanced scheduling with beta/uniform distributions.
*   **HiDream MoE:** Mixture of Experts gate loss augmentation.
*   **T5 Masked Training:** Enhanced fine details for Flux and compatible models.
*   **QKV Fusion:** Memory and speed optimizations (Flux, Lumina2).
*   **TREAD Integration:** Selective token routing for Wan and Flux models.
*   **Classifier-Free Guidance:** Optional CFG reintroduction for distilled models.

## Quickstart Guides

Get started with detailed guides for each model:

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

## Hardware Requirements

### General Requirements

*   **NVIDIA:** RTX 3080+ recommended.
*   **AMD:** 7900 XTX 24GB and MI300X verified.
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA training.

### Memory Guidelines by Model Size

*   **Large models (12B+)**: A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B)**: 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B)**: 12G+ sufficient for most training types

**Note**: Quantization (int8/fp8/nf4) significantly reduces memory requirements.

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

For more detailed installation instructions, see the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable debug logs for detailed insights: `export SIMPLETUNER_LOG_LEVEL=DEBUG`.
For performance analysis of the training loop: `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
Consult the [OPTIONS documentation](/documentation/OPTIONS.md) for available settings.