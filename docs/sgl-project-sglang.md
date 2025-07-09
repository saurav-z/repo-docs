<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)
</div>

---

## SGLang: Supercharge Your LLM and VLM Serving for Blazing-Fast Performance

SGLang is a high-performance serving framework that dramatically accelerates Large Language Model (LLM) and Vision Language Model (VLM) inference, offering both a fast backend runtime and a flexible frontend language.  [See the original repository](https://github.com/sgl-project/sglang) for more details.

### Key Features:

*   **Blazing-Fast Backend Runtime:**
    *   RadixAttention for prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation
    *   Speculative decoding
    *   Continuous batching
    *   Paged attention
    *   Tensor parallelism, pipeline parallelism, and expert parallelism
    *   Structured outputs
    *   Chunked prefill
    *   Quantization support (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching

*   **Intuitive Frontend Language:**
    *   Chained generation calls
    *   Advanced prompting
    *   Control flow
    *   Multi-modal inputs
    *   Parallelism
    *   External interactions

*   **Extensive Model Support:**
    *   Supports various generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, etc.)
    *   Supports embedding models (e5-mistral, gte, mcdse)
    *   Supports reward models (Skywork)
    *   Easy to extend for new models.

*   **Active Community and Industry Adoption:** SGLang is open-source and benefits from a vibrant community.

### Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Performance and Benchmarks

Explore the latest performance gains and benchmark results in these blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

### Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

### Adoption and Sponsorship

SGLang powers production deployments generating trillions of tokens daily, and is trusted by leading organizations across the tech industry:

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Organizations adopting SGLang" width="800" margin="10px">

### Contact Us

For enterprise inquiries regarding large-scale adoption, technical consulting, sponsorships, or partnerships, please reach out to us at [contact@sglang.ai](mailto:contact@sglang.ai).

### Acknowledgment

SGLang draws inspiration and reuses code from the following projects:
[Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).