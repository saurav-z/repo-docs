# MaxText: High-Performance, Scalable LLM Training with JAX

**Maximize your LLM training efficiency with MaxText, an open-source library leveraging JAX for cutting-edge performance and scalability on TPUs and GPUs.**  Explore state-of-the-art Large Language Models (LLMs) and streamline your research and production pipelines with MaxText's powerful capabilities. [Visit the original repository on GitHub](https://github.com/AI-Hypercomputer/maxtext).

**Key Features:**

*   **High Performance:** Achieves high Model FLOPs Utilization (MFU) and tokens/second, optimized for speed.
*   **Scalability:** Supports training on single hosts to very large clusters of TPUs and GPUs.
*   **Open-Source:** Built on pure Python/JAX, fostering community collaboration and customization.
*   **Model Library:** Provides a diverse selection of models, including Gemma, Llama, DeepSeek, Qwen, and Mistral, ready for pre-training and post-training.
*   **Pre-training & Post-training Support:** Offers comprehensive support for both pre-training and post-training techniques, including Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **Modular and Flexible:** Designed for easy experimentation, modification, and integration into your LLM projects.
*   **Comprehensive Ecosystem:** Leverages JAX AI libraries, including Flax, Tunix, Orbax, Optax, and Grain for a complete training solution.

## Installation

Get started with MaxText using either PyPI or by installing from source.

### From PyPI (Recommended)

This method offers the easiest installation for the latest stable release.

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:** The `install_maxtext_github_deps` command is required to install dependencies directly from GitHub that are not yet available on PyPI.

> **Note:** We recommend using the `--resolution=lowest` flag for `uv` to ensure a consistent and reproducible environment, critical for stable performance and benchmark repeatability.

### From Source

Install from source for access to the latest features and to contribute to the project.

```bash
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Install dependencies in editable mode
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

Verify installation with `python3 -c "import MaxText"` and run training jobs with `python3 -m MaxText.train ...`.

## 🔥 Latest News 🔥

*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
*   \[August 13, 2025] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 280B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.
*   \[July 27, 2025] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md)
*   \[July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) and provide feedback.
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
*   \[June 25, 2025] DeepSeek R1-0528 variant is now supported
*   \[April 24, 2025] Llama 4 Maverick models are now supported

## Use Cases

MaxText offers a versatile platform for both pre-training and post-training LLMs.  It empowers you to build, experiment, and deploy high-performance models at scale.

### Pre-training

Utilize MaxText as a robust reference implementation for your custom LLM projects. Customize and train your models, from small dense models (Llama 8B) to large MoEs (DeepSeek-V3) and fine-tune configurations for optimal performance.

MaxText provides opinionated implementations across various performance dimensions such as: sharding, quantization, and checkpointing.

### Post-training

Leverage MaxText's scalable framework for post-training models, whether proprietary or open-source. Utilize Tunix for your projects. For RL (like GRPO), we leverage vLLM for sampling and Pathways (soon) for multi-host.

Explore a variety of models and techniques to fine-tune the perfect model for your specific use case.

*   [SFT](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh) (Supervised Fine Tuning)
*   [GRPO](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html) (Group Relative Policy Optimization)

### Model Library

MaxText aims to provide the best OSS models, whether as a reference implementation, or to post-train and then serve with vLLM.

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

Join the MaxText community!  Get involved in discussions, and development on the [Discord Channel](https://discord.com/invite/2H9PhvTcDU). Submit feedback and feature requests, and report bugs [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).