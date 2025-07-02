<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" alt="GitHub Stars">
  </a>
  <br>
  <a href="https://pypi.org/project/sglang">
    <img src="https://img.shields.io/pypi/v/sglang" alt="PyPI">
  </a>
  <img src="https://img.shields.io/pypi/dm/sglang" alt="PyPI Downloads">
  <img src="https://img.shields.io/github/license/sgl-project/sglang.svg" alt="License">
  <img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang" alt="Closed Issues">
  <img src="https://img.shields.io/github/issues-raw/sgl-project/sglang" alt="Open Issues">
  <a href="https://deepwiki.com/sgl-project/sglang">
    <img src="https://deepwiki.com/badge.svg" alt="DeepWiki">
  </a>
</div>

---

## SGLang: The High-Performance Serving Framework for LLMs

SGLang empowers developers to build and deploy large language models (LLMs) with unparalleled speed and control, offering a fast backend runtime and flexible frontend language.

**Key Features:**

*   **Blazing-Fast Backend Runtime:**
    *   RadixAttention for efficient prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation and speculative decoding.
    *   Continuous batching and paged attention.
    *   Tensor, pipeline, and expert parallelism for scalability.
    *   Support for structured outputs, chunked prefill, and quantization (FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching.
*   **Flexible Frontend Language:**
    *   Intuitive interface for streamlined LLM application development.
    *   Chained generation calls and advanced prompting.
    *   Control flow and multi-modal input support.
    *   Parallelism and external interaction capabilities.
*   **Extensive Model Support:**
    *   Broad compatibility with generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.).
    *   Support for embedding models (e5-mistral, gte, mcdse).
    *   Reward model support (Skywork).
    *   Easy extensibility for integrating new models.
*   **Active Community & Industry Adoption:**
    *   Open-source and backed by a vibrant community.
    *   Trusted by leading enterprises and institutions, including xAI, NVIDIA, and Google Cloud, and deployed on 100,000+ GPUs worldwide.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmarks and Performance

Explore detailed performance gains and features in the following blog posts:

*   [v0.2 Blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is the backbone for LLM serving across a variety of industries. For enterprise-level adoption, contact us at contact@sglang.ai.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

## Contact Us

For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgment

We've learned from and utilized code from [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

[Back to Top](#sglangtop)