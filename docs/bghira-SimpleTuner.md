# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease

**SimpleTuner empowers you to easily train and fine-tune state-of-the-art diffusion models, offering a streamlined experience for both beginners and experienced practitioners.** ([Original Repo](https://github.com/bghira/SimpleTuner))

> **Important Note:** SimpleTuner prioritizes your data privacy; no data is sent to third parties unless you explicitly enable features like `report_to`, `push_to_hub`, or configure webhooks.

## Key Features

*   **Comprehensive Model Support:** Train a wide range of diffusion models, including Stable Diffusion XL, SD3, Flux.1, Auraflow, and more.
*   **Simplified Training:** Focus on achieving great results with sensible defaults, minimizing the need for complex configurations.
*   **Optimized Performance:** Benefit from advanced caching, memory optimization, DeepSpeed integration, and multi-GPU support for efficient training.
*   **Cutting-Edge Techniques:** Leverage techniques like TREAD, Masked Loss Training, Prior Regularization, and SNR weighting for superior performance.
*   **Model-Specific Features:** Explore features tailored to individual models, such as Kontext editing for Flux models and MoE gate loss augmentation for HiDream.
*   **Flexible Hardware Support:** Works with NVIDIA, AMD, and Apple silicon GPUs.
*   **Easy Setup:** Simple installation via pip, with CUDA, ROCm, and Apple Silicon support.

## Core Features

*   **Multi-GPU Training:** Accelerate training with distributed training across multiple GPUs.
*   **Advanced Caching:** Speed up training with image, video, and caption embeddings cached to disk.
*   **Aspect Bucketing:** Support for varied image/video sizes and aspect ratios.
*   **Memory Optimization:** Train on GPUs with limited memory (16GB+) with built-in optimizations.
*   **DeepSpeed Integration:** Utilize DeepSpeed for training large models on smaller GPUs (gradient checkpointing and optimizer state offload).
*   **Cloud Storage Support:** Train directly from cloud storage (e.g., Cloudflare R2, Wasabi S3).
*   **EMA Support:** Exponential Moving Average weights for improved stability and quality.

## Model Architecture Support

| Model                 | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders        |
| --------------------- | ---------- | --------- | ------- | ----------- | ---------- | ------------ | ------------- | -------------------- |
| Stable Diffusion XL   | 3.5B       | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L/G             |
| Stable Diffusion 3    | 2B-8B      | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L/G + T5-XXL    |
| Flux.1                | 12B        | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL      |
| Auraflow              | 6.8B       | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | UMT5-XXL             |
| PixArt Sigma          | 0.6B-0.9B  | ✗         | ✓       | ✓           | ✓          | int8         | ✗             | T5-XXL               |
| Sana                  | 0.6B-4.8B  | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2-2B            |
| Lumina2               | 2B         | ✓         | ✓       | ✓           | ✗          | int8         | ✓             | Gemma2               |
| Kwai Kolors           | 5B         | ✓         | ✓       | ✓           | ✗          | ✗            | ✗             | ChatGLM-6B           |
| LTX Video             | 5B         | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL               |
| Wan Video             | 1.3B-14B   | ✓         | ✓       | ✓*          | ✗          | int8         | ✓             | UMT5                 |
| HiDream               | 17B (8.5B MoE) | ✓         | ✓       | ✓*          | ✓          | int8/fp8/nf4 | ✓             | CLIP-L + T5-XXL + Llama |
| Cosmos2               | 2B-14B     | ✗         | ✓       | ✓           | ✗          | int8         | ✓             | T5-XXL               |
| OmniGen               | 3.8B       | ✓         | ✓       | ✓           | ✗          | int8/fp8     | ✓             | T5-XXL               |
| Qwen Image            | 20B        | ✓         | ✓       | ✓*          | ✗          | int8/nf4 (req.) | ✓             | T5-XXL               |
| SD 1.x/2.x (Legacy)   | 0.9B       | ✓         | ✓       | ✓           | ✓          | int8/nf4     | ✗             | CLIP-L               |

*✓ = Supported, ✗ = Not supported, * = Requires DeepSpeed for full-rank training*

## Advanced Training Techniques

*   **TREAD:** Token-wise dropout for transformer models.
*   **Masked Loss Training:** Improved convergence with segmentation/depth guidance.
*   **Prior Regularization:** Enhanced training stability for character consistency.
*   **Gradient Checkpointing:** Configurable intervals for memory/speed optimization.
*   **Loss Functions:** L2, Huber, Smooth L1 with scheduling support.
*   **SNR Weighting:** Min-SNR gamma weighting for improved training dynamics.

## Model-Specific Features

*   **Flux Kontext:** Edit conditioning and image-to-image training.
*   **PixArt Two-Stage:** eDiff training pipeline support.
*   **Flow Matching Models:** Advanced scheduling with beta/uniform distributions.
*   **HiDream MoE:** Mixture of Experts gate loss augmentation.
*   **T5 Masked Training:** Enhanced fine details for Flux and compatible models.
*   **QKV Fusion:** Memory and speed optimizations.
*   **TREAD Integration:** Selective token routing.
*   **Classifier-Free Guidance:** Optional CFG reintroduction.

## Quickstart Guides

Get up and running quickly with these model-specific guides:

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

*   **NVIDIA:** RTX 3080+ recommended (tested up to H200)
*   **AMD:** 7900 XTX 24GB and MI300X verified (higher memory usage vs NVIDIA)
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

*   **Large models (12B+):** A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B):** 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B):** 12G+ sufficient for most training types

**Note:** Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific details.

## Installation

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

Enable debug logs for more detailed insights:
`export SIMPLETUNER_LOG_LEVEL=DEBUG`

For performance analysis of the training loop:
`SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`

For a comprehensive list of options, see the [OPTIONS documentation](/documentation/OPTIONS.md).