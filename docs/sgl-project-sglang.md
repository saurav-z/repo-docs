<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

**SGLang: Supercharge Your LLM Performance with a High-Performance Serving Framework.**

[View the original repository on GitHub](https://github.com/sgl-project/sglang)

## Key Features

*   **Fast Backend Runtime:** Experience efficient LLM serving with features like RadixAttention, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, and more, resulting in optimized performance.
*   **Flexible Frontend Language:**  Easily program LLM applications using an intuitive interface, supporting chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
*   **Extensive Model Support:** Compatible with a wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork).
*   **Active Community:** Benefit from open-source development and a supportive community with wide industry adoption.

## What is SGLang?

SGLang is a cutting-edge serving framework designed to accelerate the performance of large language models (LLMs) and vision language models (VLMs). By co-designing its backend runtime and frontend language, SGLang offers unparalleled speed and control, making it ideal for production environments.  It is a key enabler for companies deploying AI models.

## Why Use SGLang?

*   **Boost LLM Throughput:** SGLang optimizes LLM inference, leading to faster response times and increased model utilization.
*   **Improve Model Control:** Fine-tune your LLM interactions with advanced prompting, control flow, and structured output capabilities.
*   **Seamless Model Integration:** Easily integrate and serve a wide variety of LLMs and VLMs.

## Getting Started

1.  **Installation:** [Install SGLang](https://docs.sglang.ai/start/install.html)
2.  **Quick Start:**  [Quick Start](https://docs.sglang.ai/backend/send_request.html)
3.  **Backend Tutorial:** [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
4.  **Frontend Tutorial:** [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
5.  **Contribution Guide:** [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance and Benchmarks

Explore detailed performance improvements in the following blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is the trusted LLM inference engine adopted by leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia. Deployed at scale, SGLang generates trillions of tokens daily and runs on over 1,000,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption Logo" width="800" margin="10px">

## Contact

For enterprise adoption, technical consulting, sponsorship opportunities, or partnership inquiries, please contact: contact@sglang.ai

## Acknowledgments

SGLang has been inspired by and utilizes concepts and code from: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).