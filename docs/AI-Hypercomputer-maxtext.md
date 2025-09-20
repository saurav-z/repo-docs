# MaxText: Open-Source LLM Training Library for High-Performance & Scalability

**Supercharge your Large Language Model (LLM) training with MaxText, a high-performance, open-source library built in pure Python/JAX, designed for rapid development and efficient scaling.**  [See the original repo](https://github.com/AI-Hypercomputer/maxtext).

MaxText provides a flexible and scalable foundation for LLM development, supporting a wide range of models and training techniques. Whether you're pre-training a new model or fine-tuning an existing one, MaxText delivers exceptional performance and ease of use.

**Key Features:**

*   **High-Performance & Scalable:** Optimized for Google Cloud TPUs and GPUs, achieving high Model FLOPs Utilization (MFU).
*   **Open-Source & Flexible:** Built with pure Python and JAX for easy customization and experimentation.
*   **Wide Model Support:** Includes implementations for popular models like Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Pre-training & Post-training Support:** Supports pre-training at scale and post-training techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **Comprehensive Ecosystem:** Leverages JAX AI libraries including Flax, Tunix, Orbax, Optax, and Grain for a complete training solution.
*   **Multi-Modal Training Support:** Supports multi-modal training with Gemma 3 and Llama 4 VLMs.
*   **Easy Installation:** Simple setup using `pip` with recommended use of uv package manager.

## Installation

Get started with MaxText quickly and efficiently.

### From PyPI (Recommended)

This method provides the easiest and most stable installation experience.

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

>   **Note:** The `install_maxtext_github_deps` command is temporarily needed to install dependencies directly from GitHub.

>   **Note:** The `--resolution=lowest` flag with `uv` ensures a consistent, reproducible environment by using the tested versions of dependencies defined by MaxText.

### From Source

Install from source if you need the latest features or want to contribute to MaxText.

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

## Latest News

Stay updated with the latest developments in MaxText:

*   **\[September 5, 2025]**: Repository restructured to `src` layout per [RESTRUCTURE.md](RESTRUCTURE.md).  Run `pip install -e .` from the MaxText root in existing environments.
*   **\[August 13, 2025]**: Added support for the Qwen3 2507 MoE family of models (235B Thinking, 280B Coder) and existing dense models.
*   **\[July 27, 2025]**: Updated TFLOPS/s calculations.  See [Performance Metrics](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md).
*   **\[July 16, 2025]**: Repository restructuring is proposed. Review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md).
*   **\[July 11, 2025]**: Multi-Token Prediction (MTP) training support added, inspired by the [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1).
*   **\[June 25, 2025]**: DeepSeek R1-0528 variant is now supported.
*   **\[April 24, 2025]**: Llama 4 Maverick models are now supported.

## Use Cases

MaxText empowers you to build and train cutting-edge LLMs.

### Pre-training

Use MaxText as a reference implementation for experimenting and building models from scratch. MaxText provides opinionated implementations for optimal performance, including sharding, quantization, and checkpointing.

### Post-training

Fine-tune your models with MaxText's scalable framework using Tunix. Explore various model and technique combinations to optimize performance.

*   [SFT](https://github.com/AI-Hypercomputer/maxtext/blob/main/end_to_end/tpu/llama3.1/8b/run_sft.sh)
*   [GRPO](https://maxtext.readthedocs.io/en/latest/tutorials/grpo.html)

### Model Library

MaxText provides implementations for leading open-source models.

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
    *   See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)

## Get Involved

Join the MaxText community and contribute to the project.

*   [Discord Channel](https://discord.com/invite/2H9PhvTcDU)
*   [File a feature request, documentation request, or bug report](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)