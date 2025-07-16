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

## SGLang: Supercharge Your LLMs with High-Performance Serving

**SGLang is a high-performance serving framework designed to accelerate large language models (LLMs), enabling faster and more controllable interactions with cutting-edge AI.**  [Explore the original repository](https://github.com/sgl-project/sglang).

### Key Features

*   **Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation and speculative decoding.
    *   Continuous batching and paged attention.
    *   Tensor, pipeline, and expert parallelism support.
    *   Structured outputs and chunked prefill capabilities.
    *   Quantization support: FP8/INT4/AWQ/GPTQ.
    *   Multi-LoRA batching for efficient resource usage.
*   **Flexible Frontend Language:**
    *   Intuitive interface for programming LLM applications.
    *   Supports chained generation calls and advanced prompting.
    *   Control flow, multi-modal inputs, and parallelism.
    *   External interactions for enhanced capabilities.
*   **Extensive Model Support:**
    *   Broad compatibility with generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.).
    *   Supports embedding models (e5-mistral, gte, mcdse).
    *   Reward model compatibility (Skywork).
    *   Easy extensibility for integrating new models.
*   **Active Community & Industry Adoption:**
    *   Open-source with a vibrant community.
    *   Deployed at scale, generating trillions of tokens daily.

### Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Performance & Benchmarks

Explore the latest performance improvements and benchmarks in the following blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

### Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

### Adoption and Sponsorship

SGLang is a leading open-source LLM inference engine trusted by major organizations.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

### Contact Us

For inquiries regarding enterprise adoption, technical consulting, or partnerships, please contact us at contact@sglang.ai.

### Acknowledgment

We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).