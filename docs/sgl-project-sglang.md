<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

# SGLang: High-Performance LLM Serving Framework

**SGLang empowers faster and more efficient Large Language Model (LLM) serving, making it a leading choice for production deployments.**  [Visit the original repository](https://github.com/sgl-project/sglang)

## Key Features of SGLang:

*   **Blazing-Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation
    *   Speculative decoding for improved speed
    *   Continuous batching
    *   Paged attention
    *   Tensor, pipeline, and expert parallelism
    *   Structured outputs support
    *   Chunked prefill
    *   Quantization (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching
*   **Flexible Frontend Language:**
    *   Intuitive interface for programming LLM applications
    *   Chained generation calls
    *   Advanced prompting capabilities
    *   Control flow mechanisms
    *   Support for multi-modal inputs
    *   Parallel processing options
    *   External interaction capabilities
*   **Extensive Model Support:**
    *   Wide range of generative models supported (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.)
    *   Support for embedding models (e5-mistral, gte, mcdse)
    *   Reward model compatibility (Skywork)
    *   Easy extensibility for integrating new models
*   **Active Community & Industry Adoption:**
    *   Open-source with strong community backing
    *   Used by leading enterprises and institutions

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance & Benchmarks

Explore SGLang's performance advantages and benchmark results in the following blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is deployed at scale and trusted by industry leaders, generating trillions of tokens daily, including:

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact

For enterprise adoption, technical consulting, sponsorships, or partnership inquiries, contact us at contact@sglang.ai.

## Acknowledgments

SGLang is inspired by and incorporates concepts from these projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).