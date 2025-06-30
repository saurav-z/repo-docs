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

# SGLang: High-Performance Serving Framework for LLMs and Vision Language Models

**SGLang dramatically accelerates LLM serving, empowering faster and more controllable interactions with large language models.**  [Learn more about SGLang](https://github.com/sgl-project/sglang).

## Key Features

*   **Blazing-Fast Backend Runtime:**
    *   RadixAttention for efficient prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation
    *   Speculative decoding
    *   Continuous batching and paged attention
    *   Tensor, pipeline, and expert parallelism
    *   Structured outputs and chunked prefill
    *   Quantization support (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching

*   **Flexible Frontend Language:**
    *   Intuitive interface for programming LLM applications
    *   Chained generation calls and advanced prompting
    *   Control flow and multi-modal inputs
    *   Parallelism for enhanced performance
    *   External interactions for real-world applications

*   **Extensive Model Support:**
    *   Wide range of generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, and more
    *   Embedding models: e5-mistral, gte, mcdse
    *   Reward models: Skywork
    *   Easy extensibility to integrate new models

*   **Active Community and Industry Adoption:**
    *   Open-source with strong community backing
    *   Deployed at scale, serving trillions of tokens daily
    *   Trusted by leading enterprises and institutions, including xAI, NVIDIA, AMD, Google Cloud, and others.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance and Benchmarks

Explore performance improvements and benchmark results in the following blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is adopted by a wide range of leading enterprises and institutions.  It is the de facto standard in the industry, with production deployments running on over 100,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact

For inquiries regarding enterprise adoption, technical consulting, sponsorship opportunities, or partnerships, please contact us at contact@sglang.ai.

## Acknowledgment

SGLang draws inspiration and reuses code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).