<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" alt="GitHub Stars">
  </a>
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)](https://pypi.org/project/sglang)
[![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

## SGLang: Supercharge Your LLM & VLM Serving with High-Performance Efficiency

SGLang is a high-performance serving framework designed to accelerate and optimize the deployment of large language models (LLMs) and vision language models (VLMs).  [View the original repository](https://github.com/sgl-project/sglang) for more details.

### Key Features:

*   **Blazing-Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation for faster processing.
    *   Speculative decoding.
    *   Continuous batching.
    *   Paged attention.
    *   Tensor, pipeline, and expert parallelism for distributed processing.
    *   Support for structured outputs.
    *   Chunked prefill for improved efficiency.
    *   Quantization support (FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching.

*   **Intuitive Frontend Language:**
    *   Chained generation calls for complex interactions.
    *   Advanced prompting capabilities.
    *   Control flow for flexible execution.
    *   Multi-modal input support.
    *   Parallelism for concurrent operations.
    *   External interaction capabilities.

*   **Extensive Model Compatibility:**
    *   Supports a wide range of generative models, including Llama, Gemma, Mistral, Qwen, and DeepSeek.
    *   Supports embedding models, like e5-mistral, gte, and mcdse.
    *   Supports reward models (e.g., Skywork).
    *   Easy integration of new models.

*   **Active Community & Industry Adoption:**
    *   Open-source project with strong community backing.
    *   Deployed at scale, processing trillions of tokens daily.
    *   Trusted by leading enterprises and institutions.

### Getting Started:

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Performance & Benchmarks:

*   Read the release blogs to learn about the latest performance improvements and benchmark results: [v0.2](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).

### Roadmap:

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

### Adoption and Sponsorship

SGLang is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia. As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 1,000,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

### Contact Us

For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

### Acknowledgment

We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).