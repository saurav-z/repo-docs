# MaxText: High-Performance, Scalable LLM Training in JAX

**Maximize your LLM training efficiency with MaxText, an open-source library built for speed and scalability using JAX, targeting Google Cloud TPUs and GPUs.  [Learn more at the original repo](https://github.com/AI-Hypercomputer/maxtext).**

## Key Features

*   **High Performance:** Achieves high Model FLOPs Utilization (MFU) and tokens/second.
*   **Scalable Training:** Supports pre-training and post-training across thousands of chips.
*   **Open Source:** Built on pure Python/JAX, fostering community contributions.
*   **Model Support:** Includes popular models like Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Flexible Training:** Supports pre-training, Supervised Fine-Tuning (SFT), and Group Relative Policy Optimization (GRPO).
*   **JAX-Powered:** Leverages JAX and XLA compiler for optimization-free performance.
*   **Comprehensive:** Integrates with Flax, Tunix, Orbax, Optax, and Grain for end-to-end training.
*   **Multi-Modal Support:** Supports multi-modal training with Gemma 3 and Llama 4 VLMs.

## Installation

We recommend using a Python virtual environment.

### From PyPI (Recommended)

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

> **Note:** The `install_maxtext_github_deps` command is temporarily required to install dependencies directly from GitHub. Use `--resolution=lowest` for a consistent environment.

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

Verify with `python3 -c "import MaxText"` and run training jobs with `python3 -m MaxText.train ...`.

## Latest News

*   \[September 5, 2025] MaxText has moved to an `src` layout. Run `pip install -e .` from the root.
*   \[August 13, 2025] Support for Qwen3 2507 MoE models.
*   \[July 27, 2025] Updated TFLOPS/s calculation.
*   \[July 16, 2025] Repository restructuring (see [RESTRUCTURE.md](RESTRUCTURE.md)).
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support.
*   \[June 25, 2025] DeepSeek R1-0528 variant support.
*   \[April 24, 2025] Llama 4 Maverick models support.

## Use Cases

MaxText is designed for:

*   **Pre-training:** Build and train your LLMs from scratch, using MaxText as a reference implementation.
*   **Post-training:** Fine-tune your models using SFT or RL techniques like GRPO.

### Models Available

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

## Get Involved

Join the [Discord Channel](https://discord.com/invite/2H9PhvTcDU) and submit feedback, feature requests, or bug reports [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose).