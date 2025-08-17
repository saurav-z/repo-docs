<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

# SGLang: High-Performance LLM Serving Framework for Speed and Control

**SGLang empowers developers to build and deploy large language model (LLM) applications with unparalleled speed, efficiency, and control.** ([View on GitHub](https://github.com/sgl-project/sglang))

## Key Features

*   **Blazing Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler and prefill-decode disaggregation.
    *   Speculative decoding and continuous batching for optimal throughput.
    *   Paged attention, and tensor/pipeline/expert/data parallelism for scalability.
    *   Support for structured outputs, chunked prefill, and various quantization methods (FP4/FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching for efficient handling of multiple LoRA adapters.

*   **Flexible Frontend Language:**
    *   Intuitive interface for programming LLM applications.
    *   Chained generation calls for complex workflows.
    *   Advanced prompting capabilities for tailored responses.
    *   Control flow, multi-modal input support, parallelism, and external interaction capabilities.

*   **Extensive Model Support:**
    *   Supports a wide array of generative models, including Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral, and more.
    *   Compatibility with various embedding models (e5-mistral, gte, mcdse) and reward models (Skywork).
    *   Easy extensibility to integrate new models.

*   **Thriving Community & Wide Adoption:**  SGLang is an open-source project with an active community, widely adopted by industry leaders and academic institutions.

## Getting Started

*   [Installation Guide](https://docs.sglang.ai/get_started/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/basic_usage/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/basic_usage/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/references/frontend/frontend_tutorial.html)
*   [Contribution Guide](https://docs.sglang.ai/developer_guide/contribution_guide.html)

## Benchmarks and Performance

Explore detailed performance insights in the following blog posts: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is a trusted LLM inference engine, deployed at scale by numerous leading organizations.  It's used to generate trillions of tokens daily and is a de facto standard for open source LLM inference.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact Us

For enterprise adoption, technical consulting, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgment

SGLang draws inspiration and reuses code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).