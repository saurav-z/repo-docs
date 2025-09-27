# SimpleTuner: Train Powerful Diffusion Models with Ease 🚀

**SimpleTuner empowers you to fine-tune cutting-edge diffusion models with simplicity and efficiency, producing high-quality results.**

Explore the possibilities of creating and customizing diffusion models, including Stable Diffusion, with this open-source project.  With a focus on user-friendliness, SimpleTuner is designed to get you up and running quickly with state-of-the-art training techniques.  Learn more and contribute at the [original repo](https://github.com/bghira/SimpleTuner).

**Key Features:**

*   **Easy to Use:** SimpleTuner is designed with a straightforward approach to make training more accessible, with sensible defaults.
*   **Broad Model Support:** Compatible with a wide range of diffusion models, including SDXL, SD3, Flux.1, and many more (see table below).
*   **Versatile Training Options:** Supports LoRA, Lycoris, full-rank, ControlNet, quantization, and flow matching for flexible customization.
*   **Optimized Performance:** Includes advanced caching, aspect bucketing, memory optimization, DeepSpeed integration, and S3 training for efficient training.
*   **Cutting-Edge Techniques:** Offers advanced techniques like TREAD, masked loss training, prior regularization, and SNR weighting for improved model quality.
*   **Comprehensive Documentation:** Provides detailed quickstart guides and documentation to get you started quickly.
*   **Hardware Flexibility:** Supports NVIDIA, AMD, and Apple Silicon GPUs.
*   **Privacy Focused:**  No data is sent to third parties except through opt-in features or manually configured webhooks.

## Key Features in Detail

### Core Training Features

*   Multi-GPU training
*   Advanced caching
*   Aspect bucketing
*   Memory optimization
*   DeepSpeed integration
*   S3 training
*   EMA support

### Model Architecture Support

| Model | Parameters | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantization | Flow Matching | Text Encoders |
|---|---|---|---|---|---|---|---|---|
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
*   Loss functions
*   SNR weighting

### Model-Specific Features

*   Flux Kontext
*   PixArt two-stage
*   Flow matching models support
*   HiDream MoE (Mixture of Experts)
*   T5 masked training
*   QKV fusion
*   TREAD integration
*   Classifier-free guidance

## Quickstart Guides

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

*   **NVIDIA:** RTX 3080+ (tested up to H200) recommended
*   **AMD:** 7900 XTX 24GB and MI300X verified (higher memory usage vs NVIDIA)
*   **Apple:** M3 Max+ with 24GB+ unified memory for LoRA training

### Memory Guidelines by Model Size

*   **Large models (12B+):** A100-80G for full-rank, 24G+ for LoRA/Lycoris
*   **Medium models (2B-8B):** 16G+ for LoRA, 40G+ for full-rank training
*   **Small models (<2B):** 12G+ sufficient for most training types

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

For manual installation or development setup, see the [installation documentation](/documentation/INSTALL.md).

## Troubleshooting

Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment (`config/config.env`).
For performance analysis, set `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
For a complete list of options, see the [options documentation](/documentation/OPTIONS.md).