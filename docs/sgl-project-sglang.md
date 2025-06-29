<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

---

## **SGLang: The High-Performance Framework for Serving Large Language and Vision-Language Models**

SGLang is a cutting-edge framework designed to accelerate the performance of large language models (LLMs) and vision-language models (VLMs), providing a fast and efficient serving environment.  Explore the original repository [here](https://github.com/sgl-project/sglang).

**Key Features:**

*   **Blazing-Fast Backend Runtime:**
    *   RadixAttention for efficient prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation.
    *   Speculative decoding.
    *   Continuous batching.
    *   Paged attention.
    *   Tensor, pipeline, and expert parallelism.
    *   Support for structured outputs.
    *   Chunked prefill.
    *   Quantization support (FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching.
*   **Intuitive Frontend Language:**
    *   Chained generation calls.
    *   Advanced prompting capabilities.
    *   Control flow mechanisms.
    *   Multi-modal input support.
    *   Parallelism features.
    *   External interaction capabilities.
*   **Extensive Model Compatibility:**
    *   Broad support for generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, etc.).
    *   Support for embedding models (e5-mistral, gte, mcdse).
    *   Reward model compatibility (Skywork).
    *   Easy integration for new models.
*   **Active Community & Industry Adoption:**
    *   Open-source with a thriving community.
    *   Trusted and utilized by leading enterprises and institutions.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance & Benchmarks

Refer to the following blog posts for detailed performance analysis and benchmarks:
*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang powers trillions of tokens daily and is deployed at scale by industry leaders.  We are trusted and adopted by xAI, NVIDIA, AMD, Google Cloud, Oracle Cloud, LinkedIn, and more. As a leading open-source LLM inference engine, SGLang has become the de facto standard in the industry, with production deployments running on over 100,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption by Industry Leaders" width="800" margin="10px">

## Contact Us

For enterprise adoption, technical consulting, sponsorship, and partnership inquiries, please contact us at: contact@sglang.ai.

## Acknowledgment

We are grateful for the contributions and inspirations from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).