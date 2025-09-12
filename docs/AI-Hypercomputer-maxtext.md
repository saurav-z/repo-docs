# MaxText: High-Performance, Scalable LLM Training in JAX

**MaxText is an open-source library and reference implementation empowering researchers and developers to train and fine-tune large language models (LLMs) with unparalleled efficiency and scalability using JAX.** ([Original Repo](https://github.com/AI-Hypercomputer/maxtext))

## Key Features:

*   **High-Performance LLM Training:** Achieve exceptional Model FLOPs Utilization (MFU) and tokens/second on Google Cloud TPUs and GPUs.
*   **Open-Source & Flexible:** Built in pure Python and JAX, allowing for easy customization and experimentation.
*   **Scalable Training:** Supports pre-training and post-training of LLMs, from single hosts to massive clusters.
*   **Wide Model Support:** Offers a diverse library of models including Gemma, Llama, DeepSeek, Qwen, and Mistral, with more on the way.
*   **Post-Training Capabilities:** Supports Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) for fine-tuning models.
*   **Cutting-Edge Techniques:** Implements the latest advancements, including Multi-Token Prediction (MTP), for efficient training.
*   **Comprehensive Tooling:** Leverages Flax, Tunix, Orbax, Optax, and Grain for a streamlined and powerful training experience.
*   **Multi-Modal Support:** Supports multi-modal training with Gemma 3 and Llama 4 VLMs.

## Use Cases

MaxText is designed for a variety of use cases:

*   **LLM Pre-training:** Use MaxText as a reference to build your own models from scratch, experiment with configurations, and optimize for performance.
*   **LLM Post-training:** Fine-tune existing open-source or proprietary models using MaxText's scalable framework and techniques like SFT and GRPO.

## Getting Started

*   **Read the Docs:**  [https://maxtext.readthedocs.io/en/latest/](https://maxtext.readthedocs.io/en/latest/)
*   **First Run Tutorial:** [https://maxtext.readthedocs.io/en/latest/tutorials/first\_run.html](https://maxtext.readthedocs.io/en/latest/tutorials/first_run.html)

## Model Library

MaxText supports a growing library of state-of-the-art LLMs:

*   **Google:** Gemma 3, Gemma 2, Gemma 1
*   **Alibaba:** Qwen 3 MoE, Qwen 3 Dense
*   **DeepSeek:** DeepSeek-V2, DeepSeek-V3 
*   **Meta:** Llama 4 Scout & Maverick, Llama 3.3, Llama 3.1, Llama 3.0, Llama 2
*   **Open AI:** GPT3
*   **Mistral:** Mixtral, Mistral
*   **Diffusion Models:** See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion)

## Get Involved

*   **Discord Channel:** [https://discord.com/invite/2H9PhvTcDU](https://discord.com/invite/2H9PhvTcDU)
*   **Report Issues:** [https://github.com/AI-Hypercomputer/maxtext/issues/new/choose](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)