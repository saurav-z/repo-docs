# MaxText: High-Performance, Scalable LLM Training in JAX

**Maximize your LLM training with MaxText, an open-source library and reference implementation built for speed and scalability using JAX, targeting Google Cloud TPUs and GPUs.** [Check out the original repository](https://github.com/AI-Hypercomputer/maxtext)

## Key Features of MaxText:

*   **High-Performance LLM Training:** Achieve high Model FLOPs Utilization (MFU) and tokens/second from single host to large clusters.
*   **Open Source & Flexible:** Start experimenting quickly, then adapt MaxText to meet your unique LLM training needs.
*   **Built with JAX:** Leveraging the power of JAX and the XLA compiler for optimized performance.
*   **Wide Model Support:** Train and fine-tune a variety of open-source LLMs, including Gemma, Llama, DeepSeek, Qwen, and Mistral.
*   **Scalable Training:** Supports pre-training and post-training techniques like Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
*   **Comprehensive Ecosystem:** Integrates with essential JAX AI libraries like Flax, Tunix, Orbax, Optax, and Grain.
*   **Multi-Modal Training:** Supports multi-modal training with Gemma 3 and Llama 4 VLMs.
*   **Active Development:** Stay up-to-date with the latest features and optimizations with frequent updates.

## Use Cases:

MaxText offers a robust framework for both pre-training and post-training of large language models.

### Pre-training:

*   Use MaxText as a reference implementation to build and experiment with new models.
*   Optimize your models with sharding, quantization, and checkpointing techniques.
*   Train your models with high performance and scale on TPUs or GPUs.

### Post-training:

*   Utilize MaxText with Tunix for scalable post-training.
*   Fine-tune with techniques such as SFT and GRPO.
*   Easily explore various model and technique combinations to meet your unique needs.

## Supported Models:

MaxText supports a wide range of open-source LLMs:

*   **Google:** Gemma 3 (4B, 12B, 27B), Gemma 2 (2B, 9B, 27B), Gemma 1 (2B, 7B)
*   **Alibaba:** Qwen 3 MoE 2507 (235B, 480B), Qwen 3 MoE (30B, 235B), Qwen 3 Dense (0.6B, 1.7B, 4B, 8B, 14B, 32B)
*   **DeepSeek:** DeepSeek-V2 (16B, 236B), DeepSeek-V3 0528 (671B)
*   **Meta:** Llama 4 Scout (109B) & Maverick (400B), Llama 3.3 (70B), 3.1 (8B, 70B, 405B), 3.0 (8B, 70B, 405B), Llama 2 (7B, 13B, 70B)
*   **OpenAI:** GPT3 (52k, 6B, 22B, 175B)
*   **Mistral:** Mixtral (8x7B, 8x22B), Mistral (7B)
*   **Diffusion Models:** See [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Wan 2.1, Flux, SDXL, etc)

## Stay Updated:

*   Join our [Discord Channel](https://discord.com/invite/2H9PhvTcDU)
*   Report issues or suggest features [here](https://github.com/AI-Hypercomputer/maxtext/issues/new/choose)