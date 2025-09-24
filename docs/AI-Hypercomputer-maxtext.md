# MaxText: High-Performance, Scalable LLM Training in JAX

**Maximize your LLM training with MaxText, an open-source library built for speed and scale, enabling you to train cutting-edge language models on Google Cloud TPUs and GPUs.  [Explore the MaxText Repository](https://github.com/AI-Hypercomputer/maxtext)**

MaxText provides a powerful and flexible platform for training large language models (LLMs) with a focus on performance and scalability. Built in pure Python and JAX, it leverages the power of XLA for optimized execution on Google Cloud TPUs and GPUs.

**Key Features:**

*   🚀 **High Performance:** Achieve high Model FLOPs Utilization (MFU) and tokens/second, optimized for both single-host and large-cluster training.
*   🐍 **Pure Python/JAX:** Leverages the flexibility and power of JAX for automatic differentiation, optimization, and XLA compilation.
*   ☁️ **TPU & GPU Support:** Designed for training on Google Cloud TPUs and GPUs.
*   📚 **Model Variety:** Supports a diverse range of models, including Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   🧠 **Training Flexibility:** Supports both pre-training (up to tens of thousands of chips) and post-training with techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   🌱 **Open Source & Extensible:** Start experimenting quickly and then customize MaxText to meet your specific research or production needs.
*   📦 **Easy Installation:** Simple installation via PyPI or from source.
*   📖 **Comprehensive Documentation:**  Includes a [Read The Docs site](https://maxtext.readthedocs.io/en/latest/) and detailed tutorials to get you started.

## Installation

### From PyPI (Recommended)

```bash
# Install uv, a fast Python package installer
pip install uv

# Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:** The `install_maxtext_github_deps` command may be required to install dependencies directly from GitHub that are not yet available on PyPI.

> **Note:** The `--resolution=lowest` flag is highly recommended to ensure a consistent and reproducible environment.

### From Source

```bash
# Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# Install dependencies in editable mode
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

Verify installation with `python3 -c "import MaxText"` and run training jobs using `python3 -m MaxText.train ...`.

## 🔥 Latest News 🔥

*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
*   \[August 13, 2025] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 280B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.
*   \[July 27, 2025] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md)
*   \[July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) and provide feedback.
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.
*   \[June 25, 2025] DeepSeek R1-0528 variant is now supported
*   \[April 24, 2025] Llama 4 Maverick models are now supported

## Use Cases

MaxText is a versatile tool for both pre-training and post-training LLMs. It offers a streamlined approach to building, fine-tuning, and optimizing your models.

### Pre-training

Use MaxText as a reference implementation for building LLMs from scratch. Experiment with configurations and model architectures to optimize performance on TPUs or GPUs. MaxText provides opinionated implementations for optimal performance regarding sharding, quantization, and checkpointing.

### Post-training

Leverage MaxText for scalable post-training using Tunix, supporting techniques such as Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). MaxText integrates seamlessly with frameworks like vLLM (for sampling) and Pathways (for multi-host).

### Model Library

MaxText offers a comprehensive library of state-of-the-art open-source models, providing a valuable resource for both experimentation and production deployments.

**Supported JAX models in MaxText**

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
*   **OpenAI**
    *   GPT3 (52k, 6B, 22B, 175B)
*   **Mistral**
    *   Mixtral (8x7B, 8x22B)
    *   Mistral (7B)
*   **Diffusion Models**
    *   See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

**Quickstart Guides:**

*   [SFT](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh) (Supervised Fine Tuning)
*   [GRPO](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html) (Group Relative Policy Optimization)

## Get Involved

Join the community and contribute to the development of MaxText!  Share your feedback and report any issues.

*   [Discord Channel](https://discord.com/invite/2H9PhvTcDU)
*   [File an issue](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)