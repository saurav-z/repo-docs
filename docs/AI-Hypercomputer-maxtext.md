# MaxText: High-Performance LLM Training Library

**Maximize your LLM training potential with MaxText, a powerful, open-source library for building and scaling large language models on TPUs and GPUs.**  ([See the original repository](https://github.com/AI-Hypercomputer/maxtext))

MaxText is a cutting-edge, open-source library written in pure Python/JAX, designed for high-performance, scalable Large Language Model (LLM) training. It's optimized for Google Cloud TPUs and GPUs.

**Key Features:**

*   **High Performance:** Achieves impressive Model FLOPs Utilization (MFU) and tokens/second, even on large clusters.
*   **Open Source:**  Leverage the flexibility to customize and adapt the library to meet your specific needs.
*   **Model Variety:** Supports a wide range of popular LLMs, including Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Scalable Training:** Enables pre-training (up to tens of thousands of chips) and scalable post-training using techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **JAX-Powered:** Utilizes the power of JAX and the XLA compiler for "optimization-free" performance.
*   **Easy to Get Started:** Provides clear documentation and tutorials for both research and production use cases.
*   **Multi-modal Support:** Supports multi-modal training for models like Gemma 3 and Llama 4 VLMs.

## Installation

Get started quickly by installing MaxText within a Python virtual environment.

### From PyPI (Recommended)

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:** The `install_maxtext_github_deps` command is temporarily required to install dependencies directly from GitHub that are not yet available on PyPI.

> **Note:** We highly recommend the `--resolution=lowest` flag. It instructs `uv` to install the specific, tested versions of dependencies defined by MaxText, rather than the latest available ones. This ensures a consistent and reproducible environment, which is critical for stable performance and for running benchmarks.

### From Source

```bash
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Install dependencies in editable mode
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

After installation, verify with `python3 -c "import MaxText"` and run training jobs with `python3 -m MaxText.train ...`.

## 🔥 Latest News 🔥

*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
*   \[August 13, 2025] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 280B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.
*   \[July 27, 2025] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md)
*   \[July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) and provide feedback.
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
*   \[June 25, 2025] DeepSeek R1-0528 variant is now supported
*   \[April 24, 2025] Llama 4 Maverick models are now supported

## Use Cases

MaxText offers a flexible framework for both pre-training and post-training of LLMs.

*   **Pre-training:** Use MaxText as a reference implementation for building your own models, experimenting with configurations, and optimizing performance.
*   **Post-training:**  Fine-tune pre-trained models using techniques like SFT and GRPO with a scalable framework using Tunix.

### Model Library

MaxText offers a growing library of supported models.

**Supported JAX models in MaxText:**

*   **Google**
    *   Gemma 3 (4B, 12B, 27B)
    *   Gemma 2 (2B, 9B, 27B)
    *   Gemma 1 (2B, 7B)
*   **Alibaba**
    *   Qwen 3 MoE 2507 (235B, 480B)
    *   Qwen 3 MoE (30B, 235B)
    *   Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
*   **DeepSeek**
    *   DeepSeek-V2 (16B, 236B)
    *   DeepSeek-V3 0528 (671B)
*   **Meta**
    *   Llama 4 Scout (109B) & Maverick (400B)
    *   Llama 3.3 70B, 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B)
    *   Llama 2 (7B, 13B, 70B)
*   **Open AI**
    *   GPT3 (52k, 6B, 22B, 175B)
*   **Mistral**
    *   Mixtral (8x7B, 8x22B)
    *   Mistral (7B)
*   **Diffusion Models**
    *   See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

## Get Involved

Join our [Discord Channel](https://discord.com/invite/2H9PhvTcDU) and contribute by filing feature requests, documentation requests, or bug reports [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).