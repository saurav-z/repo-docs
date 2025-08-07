<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

## SGLang: High-Performance LLM Serving for Scalable AI Applications

SGLang is a blazing-fast, open-source serving framework designed to accelerate Large Language Models (LLMs) and Vision Language Models (VLMs), offering unparalleled speed and control over your AI interactions. [**Explore the SGLang Repository**](https://github.com/sgl-project/sglang)

**Key Features and Benefits:**

*   **Unmatched Performance:** SGLang utilizes cutting-edge techniques for optimal LLM serving, including RadixAttention, zero-overhead CPU scheduling, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert parallelism, structured outputs, and more.
*   **Intuitive Frontend Language:** Develop LLM applications with ease using a flexible frontend language that supports chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external integrations.
*   **Extensive Model Support:** SGLang seamlessly integrates with a wide variety of LLMs (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse), and reward models (Skywork), ensuring broad compatibility and ease of use.
*   **Active and Growing Community:** Benefit from a vibrant open-source community that drives innovation and provides robust support.
*   **Industry Adoption:** Trusted by leading enterprises and institutions across North America and Asia.

## Key Advantages

*   **Speed:** Experience faster inference with SGLang's optimized runtime.
*   **Efficiency:** Maximize resource utilization through advanced features.
*   **Control:** Fine-tune your LLM interactions with a powerful frontend language.

## Getting Started

*   [Installation](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmarks and Performance

*   **Latest Release Blogs:** Explore the release blogs for detailed performance metrics and insights:  [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is deployed at a large scale, generating trillions of tokens daily in production. It is trusted and adopted by industry leaders including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, and many more.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption Image" width="800" margin="10px">

## Contact Us

For technical consulting, sponsorship opportunities, or partnership inquiries: contact@sglang.ai

## Acknowledgments

SGLang's design incorporates concepts and code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).