# SimpleTuner: Train Cutting-Edge Diffusion Models with Ease 💹

> **Unlock the power of AI image generation!** SimpleTuner simplifies the process of training and fine-tuning diffusion models, making it accessible for everyone.

[View the SimpleTuner Repository on GitHub](https://github.com/bghira/SimpleTuner)

**SimpleTuner** is designed with simplicity in mind, prioritizing easy-to-understand code. This open-source project welcomes contributions and offers a user-friendly experience for both academic and personal endeavors.

Join our community on [Discord](https://discord.gg/CVzhX7ZA) via Terminus Research Group for discussions and support.

## Key Features:

*   **Simplified Training:** Designed for ease of use with sensible default settings.
*   **Versatile Compatibility:** Supports a wide range of image and video datasets.
*   **Cutting-Edge Techniques:** Integrates proven features for optimal performance, including:
    *   Multi-GPU training
    *   Token-wise dropout techniques (TREAD)
    *   Aspect bucketing
    *   LoRA/LyCORIS training for reduced VRAM usage
    *   DeepSpeed integration for memory-constrained systems
    *   Quantized training for reduced VRAM consumption
    *   EMA (Exponential moving average) weight network
    *   Training from S3-compatible storage
    *   ControlNet model training
    *   Mixture of Experts
    *   Masked loss training
    *   Prior regularization
    *   Webhook Support
    *   Hugging Face Hub Integration
*   **Comprehensive Model Support:**
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
    *   Legacy Stable Diffusion (SD 1.5 & 2.x)

## Table of Contents

*   [Design Philosophy](#design-philosophy)
*   [Tutorial](#tutorial)
*   [Features](#features)
*   [Hardware Requirements](#hardware-requirements)
*   [Toolkit](#toolkit)
*   [Setup](#setup)
*   [Troubleshooting](#troubleshooting)

## Design Philosophy

*   **Simplicity:** Prioritizing user-friendly defaults and ease of use.
*   **Versatility:** Supporting a wide array of image quantities and aspect ratios.
*   **Effectiveness:** Focusing on features with proven efficacy.

## Tutorial

Begin your training journey by exploring the [Tutorial](/TUTORIAL.md) for essential information.

*   **Quick Start:** Get up and running fast with the [Quick Start](/documentation/QUICKSTART.md) guide.
*   **DeepSpeed:** Optimize memory usage with [DeepSpeed documentation](/documentation/DEEPSPEED.md).
*   **Distributed Training:** Configure multi-node training with [this guide](/documentation/DISTRIBUTED.md).

## Hardware Requirements

Comprehensive hardware requirements are detailed in the original README and are available by navigating to the section of the models you are interested in.

## Toolkit

Explore the associated toolkit at [the toolkit documentation](/toolkit/README.md).

## Setup

Detailed setup instructions are available in the [installation documentation](/INSTALL.md).

## Troubleshooting

Enable debug logs by adding `export SIMPLETUNER_LOG_LEVEL=DEBUG` to your environment. Analyze training loop performance with `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`. Review all available options in [this documentation](/OPTIONS.md).
```
Key improvements and explanations:

*   **SEO-Optimized Title:**  Uses "SimpleTuner" and adds keywords like "Train," "Diffusion Models," "AI Image Generation."  This is designed to show up in search engine results.
*   **One-Sentence Hook:** Immediately grabs attention and conveys the core benefit.
*   **Clear Key Features Section:** Uses bullet points to highlight key benefits and features.
*   **Well-Organized Table of Contents:**  Improved for readability and searchability.
*   **Concise Summaries:** The original text was summarized to be more direct and readable.  Unnecessary repetition was removed.
*   **Internal Links:**  Kept and improved the internal document links (Tutorial, etc.)
*   **Clearer Language:** Improved wording for better comprehension.
*   **Focus on Benefits:** Emphasis on what the software *does* for the user.
*   **Actionable Language:**  Uses verbs like "Unlock," "Explore," "Get up and running" to encourage engagement.
*   **Concise Hardware Requirements:** Simplified and emphasized that the detailed hardware info is within the original documentation.
*   **Stronger Call to Action (Discord):**  More inviting.
*   **Removed Redundancy:** Streamlined repetitive phrases.
*   **Keywords in Headings:**  Used relevant keywords in headings.
*   **Clearer Troubleshooting:** Improved the explanation of how to troubleshoot.
*   **Removed irrelevant notes:** Removed personal note about reporting data.