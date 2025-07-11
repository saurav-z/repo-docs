<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

---

## SGLang: Supercharge Your LLM Applications with High-Performance Serving

SGLang is a fast and flexible serving framework designed to accelerate your Large Language Model (LLM) and Vision Language Model (VLM) applications.  [Explore the SGLang repository](https://github.com/sgl-project/sglang).

### Key Features:

*   **Blazing Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation for optimized processing.
    *   Speculative decoding to reduce latency.
    *   Continuous batching for improved throughput.
    *   Paged attention for efficient memory management.
    *   Tensor, pipeline, and expert parallelism for scaling.
    *   Support for structured outputs for predictable results.
    *   Chunked prefill to handle large inputs.
    *   Quantization support (FP8/INT4/AWQ/GPTQ) for reduced resource usage.
    *   Multi-LoRA batching for flexible model configurations.

*   **Intuitive Frontend Language:**
    *   Chained generation calls for complex workflows.
    *   Advanced prompting capabilities for fine-grained control.
    *   Control flow statements to manage application logic.
    *   Multi-modal input support for diverse data types.
    *   Parallelism for concurrent operations.
    *   External interactions to connect with other systems.

*   **Extensive Model Support:**
    *   Broad compatibility with generative models, including Llama, Gemma, Mistral, Qwen, DeepSeek, and others.
    *   Support for embedding models such as e5-mistral, gte, and mcdse.
    *   Integration with reward models like Skywork.
    *   Easy extensibility to incorporate new models.

*   **Thriving Community:**
    *   Open-source project with a dedicated community.
    *   Backed by industry adoption and continuous development.

### Getting Started:

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Performance and Benchmarks

For detailed performance analysis and benchmarks, refer to the following blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

### Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

### Adoption and Sponsorship

SGLang powers trillions of tokens daily, trusted and deployed by leading organizations.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

### Contact Us

For enterprise adoption, technical consulting, sponsorship, or partnerships, please reach out to us at contact@sglang.ai.

### Acknowledgment

We acknowledge and appreciate the contributions of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

---