# MaxText: High-Performance, Open-Source LLM Training in JAX

**Supercharge your Large Language Model (LLM) training with MaxText, a powerful and scalable library built on JAX for Google Cloud TPUs and GPUs.**  [See the original repository](https://github.com/AI-Hypercomputer/maxtext)

MaxText empowers researchers and developers to efficiently train and fine-tune LLMs, offering state-of-the-art performance and a flexible framework for experimentation.

**Key Features:**

*   **High-Performance & Scalability:** Designed for optimal Model FLOPs Utilization (MFU) and tokens/second, from single host to large clusters.
*   **Open Source and Flexible:**  A pure Python/JAX library allowing for easy modification to suit your needs.
*   **Comprehensive Model Support:**  Supports pre-training and post-training for popular models like Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Cutting-Edge Training Techniques:** Supports Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) and Multi-Token Prediction (MTP).
*   **JAX-Powered:** Leverages the power of JAX and the XLA compiler for efficient model execution.
*   **Integrated Ecosystem:** Utilizes Flax, Tunix, Orbax, Optax, and Grain for a complete training pipeline.

## Installation

Get started by installing MaxText in a Python virtual environment.

### Recommended: From PyPI

This is the easiest way to get the latest stable version:

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:** The `install_maxtext_github_deps` command is temporarily required to install dependencies directly from GitHub that are not yet available on PyPI.

> **Note:** For consistent performance and reproducibility, we highly recommend the `--resolution=lowest` flag. It instructs `uv` to install the specific, tested versions of dependencies.

### From Source

Install from source if you plan to contribute or need the latest features:

```bash
# 1. Clone the repository
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext

# 2. Install dependencies in editable mode
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

After installation, verify the package is available with `python3 -c "import MaxText"` and run training jobs with `python3 -m MaxText.train ...`.

## 🔥 Latest News 🔥

*   **[September 5, 2025]**: MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
*   **[August 13, 2025]**: Qwen3 2507 MoE models (235B, 280B) are now supported, in addition to existing dense models.
*   **[July 27, 2025]**: Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) and attention flops calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030)).
*   **[July 16, 2025]**: Repository restructuring planned; see [RESTRUCTURE.md](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) for details and feedback.
*   **[July 11, 2025]**: Multi-Token Prediction (MTP) training support!
*   **[June 25, 2025]**: DeepSeek R1-0528 variant is now supported.
*   **[April 24, 2025]**: Llama 4 Maverick models are now supported.

## Use Cases

MaxText excels in both pre-training and post-training LLMs, providing a robust and scalable solution for various applications.

MaxText integrates JAX AI libraries and presents training at scale with Flax, Tunix, Orbax, Optax, and Grain.

### Pre-training

Use MaxText as a reference implementation and a starting point for your custom model training. Fork and modify MaxText to build and train LLMs, whether dense (Llama 8B) or MoE (DeepSeek-V3).

### Post-training

Fine-tune both proprietary and open-source models within MaxText's scalable framework, leveraging Tunix for RL like GRPO.

*   **SFT:** [SFT Guide](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh)
*   **GRPO:** [GRPO Tutorial](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html)

### Model Library

Access a diverse range of pre-trained models within MaxText, enabling you to choose and utilize them.

**Supported JAX Models:**

*   **Google:** Gemma 3 (4B, 12B, 27B), Gemma 2 (2B, 9B, 27B), Gemma 1 (2B, 7B)
*   **Alibaba:** Qwen 3 MoE 2507 (235B, 480B), Qwen 3 MoE (30B, 235B), Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
*   **DeepSeek:** DeepSeek-V2 (16B, 236B), DeepSeek-V3 0528 (671B)
*   **Meta:** Llama 4 Scout (109B) & Maverick (400B), Llama 3.3 70B, 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B), Llama 2 (7B, 13B, 70B)
*   **Open AI:** GPT3 (52k, 6B, 22B, 175B)
*   **Mistral:** Mixtral (8x7B, 8x22B), Mistral (7B)
*   **Diffusion Models:** See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

## Get Involved

Join the MaxText community on our [Discord Channel](https://discord.com/invite/2H9PhvTcDU) and contribute via feature requests, documentation requests, or bug reports [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).