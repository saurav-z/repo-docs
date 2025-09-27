# MaxText: High-Performance, Scalable LLM Training on JAX

**Unlock the power of large language models with MaxText, a cutting-edge, open-source library for efficient LLM training.**

[![MaxText Package Tests](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/RunTests.yml/badge.svg)](https://github.com/AI-Hypercomputer/maxtext/actions/workflows/build_and_test_maxtext.yml)

MaxText is a high-performance, scalable LLM library and reference implementation written in pure Python with JAX, designed for training on Google Cloud TPUs and GPUs. This powerful framework empowers researchers and developers to train and fine-tune large language models with unprecedented speed and efficiency.

**Key Features:**

*   **High Performance:** Achieves high Model FLOPs Utilization (MFU) and tokens/second.
*   **Scalable Training:** Supports pre-training and post-training across a range of hardware from single host to very large clusters.
*   **Open Source & Flexible:** Built on JAX for easy customization and experimentation.
*   **Wide Model Support:** Includes implementations for popular models like Gemma, Llama, DeepSeek, Qwen, Mistral, and GPT-OSS.
*   **Post-Training Techniques:** Supports popular techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **Comprehensive Library:** Provides a library of models and demonstrates how to perform pre-training or post-training with high performance and scale.

[Explore the MaxText Repository on GitHub](https://github.com/AI-Hypercomputer/maxtext)

## Installation

Get started quickly by installing MaxText using either PyPI or from source.

### From PyPI (Recommended)

This method provides the easiest access to the latest stable version of MaxText.

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

>   **Note:** The `install_maxtext_github_deps` command is temporarily required to install dependencies directly from GitHub that are not yet available on PyPI.
>   **Note:** We highly recommend the `--resolution=lowest` flag, as it ensures a consistent and reproducible environment.

### From Source

For those looking to contribute or access the latest features:

```bash
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Install dependencies in editable mode
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

After installation, confirm the package is available with `python3 -c "import MaxText"` and initiate training with `python3 -m MaxText.train ...`.

## 🔥 Latest News 🔥

*   \[September 24, 2025] The GPT-OSS family of models (20B, 120B) is now supported.
*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
*   \[August 13, 2025] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 280B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.
*   \[July 27, 2025] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md)
*   \[July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) and provide feedback.
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
*   \[June 25, 2025] DeepSeek R1-0528 variant is now supported.
*   \[April 24, 2025] Llama 4 Maverick models are now supported.

## Use Cases

MaxText empowers you to build and fine-tune large language models with exceptional performance and scalability.

MaxText leverages [JAX AI libraries](https://docs.jaxstack.ai/en/latest/getting_started.html) and presents a cohesive and comprehensive demonstration of training at scale by using [Flax](https://flax.readthedocs.io/en/latest/) (neural networks), [Tunix](https://github.com/google/tunix) (post-training), [Orbax](https://orbax.readthedocs.io/en/latest/) (checkpointing), [Optax](https://optax.readthedocs.io/en/latest/) (optimization), and [Grain](https://google-grain.readthedocs.io/en/latest/) (dataloading).

In addition to pure text-based LLMs, we also support multi-modal training with Gemma 3 and Llama 4 VLMs.

### Pre-training

Use MaxText as a reference implementation for your model training projects. Experiment with configurations and model designs to build the most efficient models, whether dense or MoE.

MaxText provides opinionated implementations for optimal performance, covering sharding, quantization, and checkpointing.

### Post-training

MaxText offers a scalable framework using Tunix for post-training models, whether proprietary or open-source. MaxText also provides support for RL (like GRPO), leveraging vLLM for sampling and Pathways (soon) for multi-host.

Our goal is to provide a variety of models (dimension “a”) and techniques (dimension “b”), so you can easily explore (a) \* (b) combinations and efficiently train the perfect model for your use case.

Check out these getting started guides:

*   [SFT](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh) (Supervised Fine Tuning)
*   [GRPO](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html) (Group Relative Policy Optimization)

### Model Library

MaxText offers a selection of high-quality OSS models for both reference and post-training.

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
    *   DeepSeek-V3 0324 (671B) & DeepSeek-R1 0528 (671B)
    *   DeepSeek-V2 (16B, 236B)
*   Meta
    *   Llama 4 Scout (109B) & Maverick (400B)
    *   Llama 3.3 70B, 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B)
    *   Llama 2 (7B, 13B, 70B)
*   Open AI
    *   GPT-OSS (20B, 120B)
    *   GPT3 (52K, 6B, 22B, 175B)
*   Mistral
    *   Mixtral (8x7B, 8x22B)
    *   Mistral (7B)
*   Diffusion Models
    *   See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

## Get Involved

Join the MaxText community on our [Discord Channel](https://discord.com/invite/2H9PhvTcDU). Provide feedback, submit feature requests, documentation requests, or report bugs [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).