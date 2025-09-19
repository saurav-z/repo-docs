# MaxText: High-Performance, Scalable LLM Training with JAX

**Train state-of-the-art large language models (LLMs) efficiently on TPUs and GPUs with MaxText, a flexible and open-source library.**  [See the original repository](https://github.com/AI-Hypercomputer/maxtext)

MaxText is a powerful, open-source library written in pure Python and [JAX](https://docs.jax.dev/en/latest/jax-101.html) designed for high-performance, scalable LLM training on Google Cloud TPUs and GPUs. It provides a comprehensive framework for both pre-training and post-training, supporting a wide range of models and training techniques. MaxText is your launching pad for ambitious LLM projects, offering a streamlined approach to building and optimizing your models.

**Key Features:**

*   **High Performance:** Achieves high Model FLOPs Utilization (MFU) and tokens/second, from single host to large clusters.
*   **Open Source & Flexible:** Built with pure Python and JAX, allowing for easy customization and experimentation.
*   **Wide Model Support:** Includes pre-configured models like Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Scalable Training:** Supports pre-training on tens of thousands of chips and scalable post-training.
*   **Popular Training Techniques:** Implements Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **JAX-Powered:** Leverages the power of JAX and the XLA compiler for optimization.
*   **Comprehensive Ecosystem:** Integrates with Flax, Tunix, Orbax, Optax, and Grain.

## Installation

Get started by installing MaxText within a Python virtual environment.

### From PyPI (Recommended)

This is the easiest way to install the latest stable version.

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

>   **Note:** The `install_maxtext_github_deps` command is temporarily required to install dependencies directly from GitHub.
>
>   **Note:** The `--resolution=lowest` flag is highly recommended to ensure a consistent and reproducible environment.

### From Source

For contributing or accessing the latest features.

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

MaxText empowers you to pre-train or post-train models at scale.

MaxText uses [JAX AI libraries](https://docs.jaxstack.ai/en/latest/getting_started.html) and provides a comprehensive demonstration of training at scale by using [Flax](https://flax.readthedocs.io/en/latest/) (neural networks), [Tunix](https://github.com/google/tunix) (post-training), [Orbax](https://orbax.readthedocs.io/en/latest/) (checkpointing), [Optax](https://optax.readthedocs.io/en/latest/) (optimization), and [Grain](https://google-grain.readthedocs.io/en/latest/) (dataloading).

In addition to pure text-based LLMs, we also support multi-modal training with Gemma 3 and Llama 4 VLMs.

### Pre-training

Use MaxText as a reference implementation to train your own models by forking and modifying it, whether it's a small dense model like Llama 8B, or a large MoE like DeepSeek-V3.

MaxText provides opinionated implementations for optimal performance across sharding, quantization, and checkpointing.

### Post-training

Post-train your model, whether it is proprietary or open source, MaxText provides a scalable framework using Tunix. For RL (like GRPO), we leverage vLLM for sampling and Pathways (soon) for multi-host.

Our goal is to provide a variety of models (dimension “a”) and techniques (dimension “b”), so you can easily explore (a) \* (b) combinations and efficiently train the perfect model for your use case.

Check out these getting started guides:

*   [SFT](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh) (Supervised Fine Tuning)
*   [GRPO](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html) (Group Relative Policy Optimization)

### Model Library

MaxText aims to provide you with the best OSS models, whether as a reference implementation, or to post-train and then serve with vLLM.

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

Join our [Discord Channel](https://discord.com/invite/2H9PhvTcDU) and contribute by filing feature requests, documentation requests, or bug reports [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).