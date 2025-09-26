# MaxText: High-Performance, Scalable LLM Training and Fine-tuning

**Supercharge your Large Language Model (LLM) projects with MaxText, an open-source, high-performance library for training and fine-tuning LLMs on Google Cloud TPUs and GPUs!**

[View the original repository on GitHub](https://github.com/AI-Hypercomputer/maxtext)

MaxText is a powerful and versatile library built in pure Python/JAX, designed for LLM research and production. It empowers you to:

*   **Train & Fine-tune at Scale:** Supports pre-training (up to tens of thousands of chips) and scalable post-training with techniques like SFT and GRPO.
*   **High Performance:** Achieves high Model FLOPs Utilization (MFU) and tokens/second.
*   **Open-Source & Flexible:** Open-source and easily modifiable to meet diverse needs.
*   **Broad Model Support:** Includes Gemma, Llama, DeepSeek, Qwen, Mistral, and more.
*   **JAX-Powered:** Leverages the power of JAX and the XLA compiler for efficient execution.

## Key Features

*   **Open-Source LLM Library:** Provides a comprehensive platform for LLM development.
*   **Pre-training and Post-training Support:** Enables both initial model creation and fine-tuning.
*   **High Performance on TPUs & GPUs:** Optimized for Google Cloud hardware, maximizing throughput.
*   **Model Flexibility:** Compatible with a wide range of popular LLM architectures, including Gemma, Llama, and more.
*   **User-Friendly:** Accessible for both research and production LLM projects.

## Installation

Follow these steps to get started:

### From PyPI (Recommended)

```bash
# 1. Install uv, a fast Python package installer
pip install uv

# 2. Install MaxText and its dependencies
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

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

*   \[September 24, 2025] The GPT-OSS family of models (20B, 120B) is now supported.
*   \[September 5, 2025] MaxText has moved to an `src` layout as part of [RESTRUCTURE.md](RESTRUCTURE.md). For existing environments, please run `pip install -e .` from MaxText root.
*   \[August 13, 2025] The Qwen3 2507 MoE family of models is now supported: MoEs: 235B Thinking & 280B Coder as well as existing dense models: 0.6B, 4B, 8B, 14B, and 32B.  
*   \[July 27, 2025] Updated TFLOPS/s calculation ([PR](https://github.com/AI-Hypercomputer/maxtext/pull/1988)) to account for causal attention, dividing the attention flops in half. Accounted for sliding window and chunked attention reduced attention flops in [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2009) and [PR](https://github.com/AI-Hypercomputer/maxtext/pull/2030). Changes impact large sequence configs, as explained in this [doc](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/guides/performance_metrics.md)  
*   \[July 16, 2025] We will be restructuring the MaxText repository for improved organization and clarity. Please review the [proposed structure](https://github.com/AI-Hypercomputer/maxtext/blob/main/RESTRUCTURE.md) and provide feedback.  
*   \[July 11, 2025] Multi-Token Prediction (MTP) training support\! Adds an auxiliary loss based on predicting multiple future tokens, inspired by [DeepSeek-V3 paper](https://arxiv.org/html/2412.19437v1), to enhance training efficiency.  
*   \[June 25, 2025] DeepSeek R1-0528 variant is now supported.  
*   \[April 24, 2025] Llama 4 Maverick models are now supported.

## Use Cases

MaxText accelerates LLM projects by offering:

### Pre-training

Use MaxText as a reference implementation to train your own models, customize configs, and optimize for performance.

### Post-training

Leverage MaxText and Tunix for scalable fine-tuning using techniques such as SFT and GRPO.

### Model Library

MaxText provides a comprehensive library of supported JAX models:

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

Join the MaxText community and contribute to this exciting project!

*   [Discord Channel](https://discord.com/invite/2H9PhvTcDU)
*   [File a Feature Request, Documentation Request, or Bug Report](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)