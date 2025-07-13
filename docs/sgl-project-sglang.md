# SGLang: High-Performance Serving Framework for LLMs and VLMs

**SGLang is a powerful, open-source serving framework designed to accelerate large language model (LLM) and vision language model (VLM) deployments.** Explore the original repository [here](https://github.com/sgl-project/sglang) for the full details.

<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

---

## Key Features of SGLang:

*   **Blazing-Fast Backend Runtime:** Experience efficient serving with cutting-edge features like RadixAttention for prefix caching, zero-overhead CPU scheduling, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor parallelism, pipeline parallelism, expert parallelism, structured outputs, chunked prefill, quantization (FP8/INT4/AWQ/GPTQ), and multi-lora batching.
*   **Intuitive Frontend Language:** Easily program LLM applications with a flexible interface, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
*   **Broad Model Compatibility:** Works seamlessly with a wide array of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), and easily integrates new models.
*   **Thriving Community & Adoption:** Benefit from active community support and industry adoption, with deployments generating trillions of tokens daily.

## Why Choose SGLang?

SGLang optimizes the interaction with LLMs and VLMs, offering both a performant backend and intuitive frontend language. This co-design results in faster, more controllable model serving.

## Getting Started with SGLang:

*   **Installation:** [Install SGLang](https://docs.sglang.ai/start/install.html)
*   **Quickstart:** [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   **Backend Tutorial:** [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   **Frontend Tutorial:** [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   **Contribution Guide:** [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance Benchmarks and Release Blogs:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap:

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship:

SGLang is a trusted solution adopted by leading enterprises and institutions.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption and Sponsorship" width="800" margin="10px">

## Contact Us:

For enterprise adoption, technical consulting, sponsorship, or partnerships, please contact us at contact@sglang.ai.

## Acknowledgments:

SGLang is built upon and inspired by the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).