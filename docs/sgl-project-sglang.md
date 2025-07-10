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

## SGLang: Supercharge Your LLM Performance with a High-Performance Serving Framework

SGLang is a blazing-fast serving framework that significantly accelerates Large Language Model (LLM) inference and streamlines the development of LLM-powered applications. [Visit the original repository](https://github.com/sgl-project/sglang) for more information.

**Key Features:**

*   **High-Performance Backend Runtime:** Experience efficient serving with cutting-edge techniques:
    *   RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation.
    *   Speculative decoding.
    *   Continuous batching.
    *   Paged attention.
    *   Tensor, pipeline, and expert parallelism.
    *   Structured outputs.
    *   Chunked prefill.
    *   Quantization support (FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching.
*   **Flexible Frontend Language:** Build LLM applications with an intuitive interface:
    *   Chained generation calls.
    *   Advanced prompting.
    *   Control flow.
    *   Multi-modal inputs.
    *   Parallelism.
    *   External interactions.
*   **Extensive Model Support:** Works with a broad range of models, including:
    *   Generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.)
    *   Embedding models (e5-mistral, gte, mcdse)
    *   Reward models (Skywork)
    *   Easy extensibility for integrating new models.
*   **Active Community & Industry Adoption:** SGLang is open-source and supported by a vibrant community, widely adopted by leading enterprises.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmarks and Performance

Explore the performance gains of SGLang in these release blogs:
*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap
[Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is the de facto industry standard for LLM inference, generating trillions of tokens daily, and is trusted and adopted by major organizations.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact Us

For enterprise inquiries, technical consulting, sponsorships, or partnership opportunities, please contact us at [contact@sglang.ai](mailto:contact@sglang.ai).

## Acknowledgment

SGLang builds upon and acknowledges the contributions of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).