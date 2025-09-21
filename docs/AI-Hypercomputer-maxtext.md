# MaxText: High-Performance, Open-Source LLM Training Library

**Supercharge your large language model training with MaxText, a cutting-edge, open-source library built for speed, scalability, and efficiency.** ([Original Repository](https://github.com/AI-Hypercomputer/maxtext))

MaxText is a powerful and flexible library designed for training and fine-tuning large language models (LLMs) on Google Cloud TPUs and GPUs, written in pure Python and JAX. Whether you're a researcher or a production engineer, MaxText offers a streamlined approach to building, experimenting with, and deploying state-of-the-art LLMs.

**Key Features:**

*   **High Performance:** Optimized for Model FLOPs Utilization (MFU) and tokens/second, ensuring efficient training across various hardware configurations.
*   **Scalable Architecture:** Designed to scale from single host to large clusters, supporting pre-training and post-training tasks.
*   **Open Source and Flexible:** Built on JAX, a powerful and flexible framework, and supports popular models. Easily adaptable to your custom needs.
*   **Model Agnostic:** Supports a wide range of LLMs, including:
    *   Gemma
    *   Llama
    *   DeepSeek
    *   Qwen
    *   Mistral
*   **Comprehensive Training Support:** Includes pre-training, Supervised Fine-Tuning (SFT), and Group Relative Policy Optimization (GRPO) to cover a variety of LLM use cases.

## Installation

Get started quickly by installing MaxText using your preferred method:

### From PyPI (Recommended)

```bash
pip install uv
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:**  Using `--resolution=lowest` ensures consistent and reproducible environments.

### From Source

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

Verify installation with `python3 -c "import MaxText"` and run training jobs using `python3 -m MaxText.train ...`.

## 🔥 What's New 🔥

Stay up-to-date with the latest MaxText developments:

*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md).  For existing environments, please run `pip install -e .` from MaxText root.
*   \[August 13, 2025] Support added for the Qwen3 2507 MoE family of models.
*   \[July 27, 2025] TFLOPS/s calculation updated.
*   \[July 16, 2025]  Repository restructuring proposed; see [RESTRUCTURE.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md).
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support added.
*   \[June 25, 2025] Support for DeepSeek R1-0528 variant.
*   \[April 24, 2025] Support for Llama 4 Maverick models.

## Use Cases

MaxText is your go-to solution for:

### Pre-training

Use MaxText to experiment, ideate, and train LLMs from scratch.  The library provides opinionated implementations for achieving optimal performance.

### Post-training

Fine-tune existing models using a scalable framework built on Tunix. Leverage vLLM and Pathways (coming soon) for efficient RL training (like GRPO).

### Model Library

Access a rich collection of pre-supported models:

**Supported JAX models in MaxText**

*   Google
    *   Gemma 3 (4B, 12B, 27B)
    *   Gemma 2 (2B, 9B, 27B)
    *   Gemma 1 (2B, 7B)
*   Alibaba
    *   Qwen 3 MoE 2507 (235B, 480B)
    *   Qwen 3 MoE (30B, 235B)
    *   Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
*   DeepSeek
    *   DeepSeek-V2 (16B, 236B)
    *   DeepSeek-V3 0528 (671B)
*   Meta
    *   Llama 4 Scout (109B) & Maverick (400B)
    *   Llama 3.3 70B, 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B)
    *   Llama 2 (7B, 13B, 70B)
*   Open AI
    *   GPT3 (52k, 6B, 22B, 175B)
*   Mistral
    *   Mixtral (8x7B, 8x22B)
    *   Mistral (7B)
*   Diffusion Models
    *   See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

## Get Involved

Join the MaxText community!

*   [Discord Channel](https://discord.com/invite/2H9PhvTcDU)
*   Report issues, request features, or ask questions: [Issue Tracker](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)