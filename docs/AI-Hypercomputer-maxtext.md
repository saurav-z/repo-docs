# MaxText: Train and Fine-Tune Large Language Models at Scale

**Supercharge your LLM training with MaxText, an open-source library providing high-performance and scalable solutions for cutting-edge language models.**

[![MaxText Package Tests](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/build_and_test_maxtext.yml)

## Key Features

*   **High-Performance & Scalable:** Built for speed and efficiency, achieving high Model FLOPs Utilization (MFU) and tokens/second on both single hosts and massive clusters, leveraging the power of JAX and the XLA compiler.
*   **Open Source & Flexible:** A versatile library for ambitious LLM projects, perfect for both research and production environments, readily adaptable for your unique needs.
*   **Pre-training & Post-training Support:** Comprehensive support for both pre-training and post-training, including popular techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **Extensive Model Library:** Access a wide range of models, including Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Pure Python/JAX:** Written in pure Python with JAX for flexibility and performance, targeting Google Cloud TPUs and GPUs.
*   **Modular Design:** Uses Flax for neural networks, Tunix for post-training, Orbax for checkpointing, Optax for optimization, and Grain for dataloading.

## Getting Started

### Installation

We recommend installing MaxText within a Python virtual environment.

#### From PyPI (Recommended)

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:** The `install_maxtext_github_deps` command is temporarily required to install dependencies directly from GitHub that are not yet available on PyPI.

> **Note:** We highly recommend the `--resolution=lowest` flag with `uv` for consistent and reproducible results.

#### From Source

```bash
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Install dependencies in editable mode
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

After installation, verify with `python3 -c "import MaxText"` and train with `python3 -m MaxText.train ...`.

## Latest Updates

*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md).
*   \[August 13, 2025] The Qwen3 2507 MoE family of models is now supported.
*   \[July 27, 2025] Updated TFLOPS/s calculation.
*   \[July 16, 2025] Restructuring of the MaxText repository is in progress.
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support.
*   \[June 25, 2025] DeepSeek R1-0528 variant is now supported.
*   \[April 24, 2025] Llama 4 Maverick models are now supported.

## Use Cases

MaxText provides a robust framework for both pre-training and post-training LLMs.

### Pre-training

Use MaxText as a reference implementation for experimentation and to train your own custom models, offering optimized configurations for performance across sharding, quantization, and checkpointing.

### Post-training

Leverage MaxText and Tunix for scalable post-training, including Reinforcement Learning (RL) methods like GRPO, with integration with vLLM and Pathways.

### Model Library

MaxText supports a wide range of state-of-the-art open-source language models.

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
    *   See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)

## Get Involved

Join the community on our [Discord Channel](https://discord.com/invite/2H9PhvTcDU).  For feedback, feature requests, documentation requests, or bug reports, please [submit an issue](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose) on GitHub.

[Back to the top of the repository](https://github.com/AI-Hypercomputer/maxtext)