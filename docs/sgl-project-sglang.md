<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)](https://pypi.org/project/sglang)
[![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

## SGLang: Supercharge Your LLM Serving with Speed and Control

SGLang is a high-performance serving framework designed to accelerate large language models and vision language models, offering both a fast backend runtime and a flexible frontend language for building powerful AI applications.  You can find the original repository [here](https://github.com/sgl-project/sglang).

**Key Features:**

*   **Blazing-Fast Backend Runtime:**
    *   RadixAttention for efficient prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation for optimized processing
    *   Speculative decoding to speed up generation
    *   Continuous batching for increased throughput
    *   Paged attention for memory efficiency
    *   Tensor, pipeline, and expert parallelism support
    *   Structured output capabilities
    *   Chunked prefill to handle large inputs
    *   Quantization support (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching

*   **Intuitive Frontend Language:**
    *   Chained generation calls for complex interactions
    *   Advanced prompting for better control
    *   Control flow statements to manage logic
    *   Multi-modal input handling
    *   Parallel processing for faster results
    *   External interaction capabilities for integration

*   **Extensive Model Compatibility:**
    *   Supports a wide array of generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, and more.
    *   Works with various embedding models: e5-mistral, gte, mcdse
    *   Compatible with reward models: Skywork
    *   Easy extensibility to integrate new models

*   **Active Community & Industry Adoption:**
    *   Open-source with a vibrant community
    *   Trusted by leading enterprises and institutions (xAI, NVIDIA, AMD, etc.)

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance and Benchmarks

Explore the performance improvements in these release blogs:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is powering production deployments generating trillions of tokens daily. It is the de facto standard for LLM inference in the industry, with production deployments running on over 100,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

## Contact Us

For inquiries regarding enterprise adoption, technical consulting, sponsorship opportunities, or partnerships, please contact us at contact@sglang.ai.

## Acknowledgment

We've learned from and reused code from these projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).