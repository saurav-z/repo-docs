# SimpleTuner: Train Cutting-Edge AI Models with Ease 🚀

SimpleTuner simplifies AI model training, empowering you to fine-tune a wide range of models with a focus on ease of use and understanding. **Find the original repo [here](https://github.com/bghira/SimpleTuner).**

> ℹ️ **Important:** No data is sent to third parties unless explicitly enabled via the `report_to`, `push_to_hub`, or webhook flags, which require manual configuration.

## Key Features

*   **User-Friendly Design:** SimpleTuner prioritizes ease of use, providing sensible default settings and reducing the need for complex configurations.
*   **Versatile Training:** Supports diverse image and video datasets, from small collections to massive datasets.
*   **Cutting-Edge Capabilities:** Includes the latest advancements in AI training, with tested and proven features.
*   **Multi-GPU Training:** Accelerate training with multi-GPU support.
*   **Advanced Techniques:**
    *   New token-wise dropout (TREAD) for faster training.
    *   Aspect bucketing for flexible image/video sizes.
    *   Optional EMA (Exponential Moving Average) for improved training stability.
    *   S3-compatible storage support for training directly from cloud storage.
*   **Model Support:**
    *   HiDream
    *   Flux.1
    *   Wan Video
    *   LTX Video
    *   PixArt Sigma
    *   NVLabs Sana
    *   Stable Diffusion 3
    *   Kwai Kolors
    *   Lumina2
    *   Cosmos2 Predict (Image)
    *   Qwen-Image
    *   Legacy Stable Diffusion models (SD 1.5, SD 2.x)
*   **DeepSpeed Integration:** Enable memory optimization for training large models on constrained hardware.
*   **Quantization:** Employ NF4/INT8/FP8 LoRA training for reduced VRAM consumption.
*   **ControlNet Training:** Support for full or LoRA-based ControlNet training for SDXL, SD 1.x/2.x, and Flux.
*   **Hugging Face Hub Integration:** Seamless model uploads and generation of model cards.

## Table of Contents

-   [Design Philosophy](#design-philosophy)
-   [Tutorial](#tutorial)
-   [Features](#features)
-   [Model-Specific Features](#model-specific-features)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Wan Video](#wan-video)
    -   [LTX Video](#ltx-video)
    -   [PixArt Sigma](#pixart-sigma)
    -   [NVLabs Sana](#nvlabs-sana)
    -   [Stable Diffusion 3](#stable-diffusion-3)
    -   [Kwai Kolors](#kwai-kolors)
    -   [Lumina2](#lumina2)
    -   [Cosmos2 Predict (Image)](#cosmos2-predict-image)
    -   [Qwen-Image](#qwen-image)
-   [Hardware Requirements](#hardware-requirements)
    -   [NVIDIA](#nvidia)
    -   [AMD](#amd)
    -   [Apple](#apple)
    -   [HiDream](#hidream)
    -   [Flux.1](#flux1)
    -   [Auraflow](#auraflow)
    -   [SDXL, 1024px](#sdxl-1024px)
    -   [Stable Diffusion 2.x, 768px](#stable-diffusion-2x-768px)
-   [Toolkit](#toolkit)
-   [Setup](#setup)
-   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Optimized settings with good defaults to minimize configuration.
*   **Versatility:** Capable of handling datasets of varying sizes.
*   **Cutting-Edge:** Integration of proven features.

## Model-Specific Features

SimpleTuner offers specialized training options and support for various model architectures:

**(Summarized Model Information here. Refer to the original README for full details)**

### HiDream

*   Custom ControlNet implementation
*   Memory-efficient training
*   MoEGate loss augmentation (optional)

### Flux.1

*   Double training speed with `--fuse_qkv_projections`
*   ControlNet training
*   Instruct fine-tuning for Kontext
*   Classifier-free guidance training
*   T5 attention masked training (optional)

### Wan Video

*   Text-to-Video training
*   LyCORIS, PEFT, and full tuning support

### LTX Video

*   LyCORIS, PEFT, and full tuning support

### PixArt Sigma

*   LyCORIS and full tuning
*   ControlNet training support
*   Two-stage PixArt training

### NVLabs Sana

*   LyCORIS and full tuning support

### Stable Diffusion 3

*   LoRA and full finetuning
*   ControlNet training

### Kwai Kolors

*   SDXL-based model with a ChatGLM text encoder

### Lumina2

*   LoRA, Lycoris, and full finetuning support

### Cosmos2 Predict (Image)

*   Lycoris and full-rank tuning support

### Qwen-Image

*   Lycoris, LoRA, and full-rank training support

## Tutorial

Before starting, review the [main tutorial](/TUTORIAL.md) for essential information.

For a quick start, use the [Quick Start](/documentation/QUICKSTART.md) guide.

## Hardware Requirements

### NVIDIA

*   General recommendation: 3080 and up

### AMD

*   LoRA and full-rank tuning are verified working on a 7900 XTX 24GB and MI300X.

### Apple

*   LoRA and full-rank tuning are tested to work on an M3 Max with 128G memory

**(Model-specific hardware requirements are detailed in the original README)**

## Toolkit

Refer to the [toolkit documentation](/toolkit/README.md) for details about the included toolkit.

## Setup

Detailed setup instructions can be found in the [installation documentation](/INSTALL.md).

## Troubleshooting

*   Enable debug logs by setting `export SIMPLETUNER_LOG_LEVEL=DEBUG` in your environment.
*   Analyze training loop performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`.
*   For a comprehensive list of options, consult [OPTIONS.md](/OPTIONS.md).
```

Key improvements and explanations:

*   **SEO Optimization:** Used relevant keywords throughout the README (AI model training, fine-tuning, Stable Diffusion, LoRA, etc.).  Included an SEO-friendly introduction and a concise summary of features.
*   **Clear Structure:**  Used headings, subheadings, and bullet points for improved readability and scannability.  The table of contents helps users quickly navigate the document.
*   **Concise Summaries:** The content is reorganized and shortened, removing verbose language.  Model-specific information is now summarized and links back to the original README for the details.
*   **Hook:**  A compelling one-sentence opening to grab the reader's attention.
*   **Call to Action (Implied):** The clear structure and features overview encourage exploration of the documentation.
*   **Focus on Value:** Highlights the benefits of SimpleTuner (ease of use, versatility, performance, model support).
*   **Removed Redundancy:**  Combined repetitive phrases and streamlined the writing.
*   **Hugging Face Integration:**  Mentioned the Hugging Face Hub more prominently.
*   **Clearer Hardware Requirements:** Simplified and summarized hardware information, which is crucial for users.
*   **Contextual Links:** Added internal links to the quickstart and tutorial guides.
*   **Markdown Formatting:**  The use of markdown ensures proper display on GitHub.
*   **Model-Specific Section:** Organized model-specific information in a separate section to improve clarity.