<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" alt="GitHub Stars">
  </a>
  <br>
  <a href="https://pypi.org/project/sglang">
    <img src="https://img.shields.io/pypi/v/sglang?color=blue" alt="PyPI Version">
    <img src="https://img.shields.io/pypi/dm/sglang?color=blue" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/sgl-project/sglang/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/sgl-project/sglang.svg" alt="License">
  </a>
  <a href="https://github.com/sgl-project/sglang/issues">
    <img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang" alt="Closed Issues">
    <img src="https://img.shields.io/github/issues-raw/sgl-project/sglang" alt="Open Issues">
  </a>
</div>

---

## SGLang: Accelerate Your LLM Inference and Unlock Trillions of Tokens Daily

SGLang is a high-performance serving framework designed to speed up large language model (LLM) and vision language model (VLM) inference, offering both a fast backend runtime and a flexible frontend language.

**Key Features:**

*   🚀 **Blazing Fast Backend Runtime:** Experience optimized serving with:
    *   RadixAttention for efficient prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation
    *   Speculative decoding
    *   Continuous batching
    *   Paged attention
    *   Tensor, pipeline, and expert parallelism
    *   Structured outputs
    *   Chunked prefill
    *   Quantization support (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching
*   💻 **Intuitive Frontend Language:** Simplify LLM application development with:
    *   Chained generation calls
    *   Advanced prompting
    *   Control flow mechanisms
    *   Multi-modal inputs
    *   Parallelism
    *   External interactions
*   🌐 **Broad Model Support:** Compatible with a wide range of models:
    *   Generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, and more
    *   Embedding models: e5-mistral, gte, mcdse
    *   Reward models: Skywork
    *   Easy extensibility for integrating new models
*   🤝 **Active Community & Industry Adoption:** Benefit from open-source development and widespread enterprise usage.

## Why Choose SGLang?

SGLang empowers you to build and deploy LLM-powered applications with unprecedented speed, efficiency, and control. Optimize your inference pipeline and unlock new possibilities in AI.

## Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance & Benchmarks

Explore the performance gains SGLang delivers:

*   [v0.2 Blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 Blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is trusted and deployed by leading organizations, processing trillions of tokens daily.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="SGLang Adoption" width="800" margin="10px">

## Contact Us

For enterprise inquiries, technical consulting, sponsorship, or partnerships, please contact us at [contact@sglang.ai](mailto:contact@sglang.ai).

## Acknowledgments

SGLang's design and code draw inspiration from: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

---

**[Back to Top](#sglangtop)**