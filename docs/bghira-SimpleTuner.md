# SimpleTuner: The Simple and Powerful Way to Fine-Tune Diffusion Models

**SimpleTuner empowers you to easily train and fine-tune a wide array of diffusion models, offering flexibility and performance for both beginners and experienced users.**  [Explore the original repository](https://github.com/bghira/SimpleTuner).

## Key Features

*   **Ease of Use:** Designed for simplicity with sensible defaults, minimizing the need for complex configuration.
*   **Broad Model Support:** Train a variety of diffusion models, including Stable Diffusion XL, Stable Diffusion 3, Flux.1, and many more (see detailed list below).
*   **Performance Optimization:** Includes features like multi-GPU training, advanced caching, and memory optimization to handle large datasets and complex models.
*   **Advanced Techniques:** Supports cutting-edge training methods such as TREAD, Masked Loss Training, SNR weighting, and Gradient Checkpointing.
*   **Hardware Flexibility:** Compatible with NVIDIA, AMD, and Apple Silicon GPUs.
*   **DeepSpeed Integration:** Train large models on smaller GPUs with gradient checkpointing and optimizer state offload.
*   **Cloud Training:** Support for training directly from cloud storage services like Cloudflare R2 and Wasabi S3.
*   **Regular Updates:** Constantly incorporating features that have proven efficacy, avoiding the addition of untested options.
*   **Community Support:** Join the community on Discord for help and discussions.

## Key Benefits

*   **Simplified Training:** Train diffusion models with minimal configuration.
*   **Reduced GPU Memory Needs:** Optimized for training on GPUs with limited memory.
*   **Improved Training Speed:** Achieve faster training with efficient caching and multi-GPU support.
*   **Better Image Quality:** Benefit from advanced techniques for stable and high-quality results.
*   **Model Versatility:** Compatible with a wide range of popular diffusion models.

## Key Features in Detail

### Core Training Features

*   Multi-GPU training
*   Advanced caching (Image, video, and caption embeddings cached to disk)
*   Aspect bucketing
*   Memory optimization
*   DeepSpeed integration
*   S3 training (Cloudflare R2, Wasabi S3)
*   EMA support

### Model Architecture Support

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

### Advanced Training Techniques

*   TREAD (Token-wise dropout)
*   Masked loss training
*   Prior regularization
*   Gradient checkpointing
*   Loss functions (L2, Huber, Smooth L1)
*   SNR weighting

### Model-Specific Features

*   Flux Kontext
*   PixArt two-stage
*   Flow matching model support
*   HiDream MoE
*   T5 masked training
*   QKV fusion
*   TREAD integration
*   Classifier-free guidance

### Quickstart Guides

Detailed quickstart guides are available for all supported models:

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

*   **NVIDIA:** RTX 3080+ recommended
*   **AMD:** 7900 XTX 24GB and MI300X verified
*   **Apple:** M3 Max+ with 24GB+ unified memory

### Memory Guidelines by Model Size

*   **Large models (12B+):** A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B):** 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B):** 12G+ sufficient for most training types

**Note:** Quantization (int8/fp8/nf4) significantly reduces memory requirements. See individual [quickstart guides](#quickstart-guides) for model-specific requirements.

## Installation

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

Enable debug logs for a more detailed insight by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment (`config/config.env`) file.

For performance analysis of the training loop, setting `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` will have timestamps that highlight any issues in your configuration.

For a comprehensive list of options available, consult [this documentation](/documentation/OPTIONS.md).