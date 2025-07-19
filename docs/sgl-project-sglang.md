<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

---

# SGLang: High-Performance Serving for Large Language Models

SGLang is a blazing-fast serving framework, streamlining interactions with large language models and vision language models.  [Learn more on GitHub](https://github.com/sgl-project/sglang).

## Key Features

*   **Blazing-Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation and speculative decoding.
    *   Continuous batching and paged attention.
    *   Tensor, pipeline, and expert parallelism.
    *   Structured outputs and chunked prefill.
    *   Quantization support (FP8/INT4/AWQ/GPTQ) and multi-LoRA batching.
*   **Intuitive Frontend Language:**
    *   Chained generation calls and advanced prompting.
    *   Control flow and multi-modal input support.
    *   Parallelism and external interactions.
*   **Extensive Model Support:**
    *   Supports various generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.).
    *   Supports embedding models (e5-mistral, gte, mcdse) and reward models (Skywork).
    *   Easy extensibility for integrating new models.
*   **Active Community and Industry Adoption:**  Open-source with significant industry adoption.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance and Benchmarks

SGLang offers significant performance improvements.  Explore the following blog posts for details:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is deployed at scale by leading organizations, processing trillions of tokens daily, and is the de facto industry standard LLM inference engine with deployments on over 1,000,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us

For enterprise adoption, technical consulting, sponsorships, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgment

SGLang builds upon and acknowledges the contributions of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).