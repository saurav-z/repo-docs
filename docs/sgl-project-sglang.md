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

## SGLang: Supercharge Your LLM Performance with a High-Performance Serving Framework

**SGLang** is a powerful and flexible serving framework designed to accelerate large language model (LLM) and vision language model (VLM) deployments, offering industry-leading speed and control.  [Explore the SGLang GitHub Repository](https://github.com/sgl-project/sglang)

### Key Features

*   **Blazing-Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation and speculative decoding.
    *   Continuous batching for optimized throughput.
    *   Paged attention and tensor/pipeline/expert parallelism.
    *   Support for structured outputs and chunked prefill.
    *   Quantization options: FP8/INT4/AWQ/GPTQ.
    *   Multi-LoRA batching for efficient resource utilization.
*   **Flexible Frontend Language:**
    *   Intuitive interface for streamlined LLM application development.
    *   Chained generation calls for complex workflows.
    *   Advanced prompting capabilities.
    *   Control flow mechanisms for dynamic behavior.
    *   Multi-modal input support.
    *   Parallelism for concurrent operations.
    *   External interaction capabilities.
*   **Extensive Model Support:**
    *   Broad compatibility with generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.).
    *   Support for various embedding models (e5-mistral, gte, mcdse).
    *   Reward model compatibility (Skywork).
    *   Easy extensibility to integrate new models.

### News & Updates

*   **[2025/06]** Awarded Open Source AI Grant by a16z.
*   **[2025/06]** 2.7x Higher Decoding Throughput on GB200 NVL72.
*   **[2025/05]** Deploying DeepSeek with PD and Large-scale Expert Parallelism.
*   **[2025/03]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X.
*   **[2025/03]** Joins PyTorch Ecosystem.
*   **[2024/12]** v0.4 Release: Zero-Overhead Batch Scheduler and more.
*   **[2024/07]** v0.2 Release: Faster Llama3 Serving.

[See More News](https://github.com/sgl-project/sglang#news)

### Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Benchmark & Performance

Explore SGLang's performance gains in these blog posts: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

### Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

### Adoption and Sponsorship

SGLang is a proven solution deployed at scale, trusted by leading enterprises and institutions, generating trillions of tokens daily.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

### Contact Us

For enterprise adoption, technical consulting, or sponsorship inquiries, contact us at [contact@sglang.ai](mailto:contact@sglang.ai).

### Acknowledgment

SGLang builds upon and acknowledges the contributions of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).