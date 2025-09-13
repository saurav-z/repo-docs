# MaxText: High-Performance, Open-Source LLM Training with JAX

**Looking to train large language models at scale? MaxText, a cutting-edge, open-source library built on JAX, empowers you to achieve unparalleled performance and scalability for your LLM projects.** [Learn more at the MaxText GitHub repo](https://github.com/AI-Hypercomputer/maxtext).

## Key Features:

*   **High Performance & Scalability:** MaxText leverages the power of JAX and the XLA compiler to deliver exceptional Model FLOPs Utilization (MFU) and tokens/second, scaling seamlessly from single hosts to massive clusters.
*   **Open Source and Flexible:** Built in pure Python/JAX, MaxText is open-source, enabling customization and experimentation for both research and production use cases.
*   **Wide Model Support:** Train and fine-tune a diverse range of models, including Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Pre-training and Post-training:** Supports both pre-training (up to tens of thousands of chips) and scalable post-training techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **Optimization-Free Design:** Simplifies LLM training while maintaining top-tier performance, thanks to the efficiency of JAX and XLA.
*   **Integration with JAX Ecosystem:** Leverages powerful JAX AI libraries, including Flax, Tunix, Orbax, Optax, and Grain, for a cohesive and comprehensive training experience.

## Installation

Choose your preferred installation method:

### Recommended: Install from PyPI

```bash
pip install uv
uv pip install maxtext --resolution=lowest
install_maxtext_github_deps
```

### Install from Source
```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
pip install uv
uv pip install -e . --resolution=lowest
install_maxtext_github_deps
```

## Use Cases

### Pre-training

Utilize MaxText as a reference implementation to train models from scratch, experimenting with configurations and model designs to optimize performance on TPUs and GPUs.

### Post-training

Leverage MaxText's scalable framework with Tunix for efficient post-training of both proprietary and open-source models.

### Model Library

MaxText supports a variety of models for pre-training and post-training:

*   **Google:** Gemma 3, Gemma 2, Gemma 1
*   **Alibaba:** Qwen 3 MoE and Dense Models
*   **DeepSeek:** DeepSeek-V2, DeepSeek-V3
*   **Meta:** Llama 4 Scout & Maverick, Llama 3, Llama 2
*   **OpenAI:** GPT3
*   **Mistral:** Mixtral, Mistral
*   **Diffusion Models:** (See MaxDiffusion)

## Get Involved

Join the MaxText community and contribute to the project:

*   [Discord Channel](https://discord.com/invite/2H9PhvTcDU)
*   [File Issues](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose) for feature requests, documentation improvements, and bug reports.