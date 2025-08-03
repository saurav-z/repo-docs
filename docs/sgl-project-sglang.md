<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

# SGLang: High-Performance LLM Serving Framework

**SGLang is a cutting-edge serving framework for large language models (LLMs) and vision language models (VLMs), designed to deliver unparalleled speed and control for your AI applications.** ([Back to original repository](https://github.com/sgl-project/sglang))

## Key Features

*   **Blazing-Fast Backend Runtime:** Experience efficient serving with advanced features such as:
    *   RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation.
    *   Speculative decoding and continuous batching.
    *   Paged attention and tensor/pipeline/expert parallelism.
    *   Structured outputs and chunked prefill.
    *   Quantization support (FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching.

*   **Flexible Frontend Language:** Simplify LLM application development with an intuitive interface:
    *   Chained generation calls.
    *   Advanced prompting capabilities.
    *   Control flow mechanisms.
    *   Multi-modal input support.
    *   Parallelism for enhanced performance.
    *   External interaction capabilities.

*   **Extensive Model Support:** Seamlessly integrate a wide array of models:
    *   Generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, etc.
    *   Vision Language Models: LLaVA and more.
    *   Embedding models: e5-mistral, gte, mcdse.
    *   Reward models: Skywork.
    *   Easy extensibility to integrate new models.

*   **Active Community & Open Source:** Benefit from an active and supportive open-source community and industry adoption.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance & Benchmarks

Explore SGLang's superior performance in the following release blogs:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is a trusted and adopted LLM inference engine, deployed at scale and generating trillions of tokens daily. It is used by major technology organizations across North America and Asia.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact Us

For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgments

SGLang is inspired by and built upon the foundations of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).