<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

# SGLang: Supercharge Your LLM Performance with a High-Performance Serving Framework

SGLang is a blazing-fast serving framework designed to accelerate large language models and vision language models, providing both efficient performance and enhanced control.  [Explore the SGLang repository on GitHub](https://github.com/sgl-project/sglang).

## Key Features

*   **Blazing-Fast Backend Runtime:**
    *   **RadixAttention for Prefix Caching:** Efficiently handles long context windows.
    *   **Zero-Overhead CPU Scheduler:** Optimizes resource utilization.
    *   **Prefill-Decode Disaggregation:** Enhances throughput during inference.
    *   **Speculative Decoding:** Accelerates text generation.
    *   **Continuous Batching:** Improves overall system efficiency.
    *   **Paged Attention:** Optimized for memory usage.
    *   **Tensor, Pipeline, Expert, and Data Parallelism:** Enables scaling across multiple GPUs.
    *   **Structured Outputs:** Simplifies complex data handling.
    *   **Chunked Prefill:** Optimizes initial processing stages.
    *   **Quantization Support (FP4/FP8/INT4/AWQ/GPTQ):** Reduces model size and improves performance.
    *   **Multi-LoRA Batching:** Supports efficient handling of multiple LoRA models.

*   **Flexible Frontend Language:**
    *   **Chained Generation Calls:** Streamlines complex workflows.
    *   **Advanced Prompting:** Offers fine-grained control over model behavior.
    *   **Control Flow:** Enables dynamic logic within prompts.
    *   **Multi-Modal Inputs:** Supports diverse input formats.
    *   **Parallelism:** Leverages concurrent execution for faster results.
    *   **External Interactions:** Facilitates integration with external services.

*   **Extensive Model Support:**
    *   Supports a wide array of LLMs, including Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral, and more.
    *   Includes support for various embedding models (e5-mistral, gte, mcdse).
    *   Offers integration with reward models (Skywork).
    *   Designed for easy integration of new models.

*   **Active Community & Wide Adoption:**
    *   Open-source with a thriving and supportive community.
    *   Trusted by numerous leading organizations and research institutions.

## Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start Tutorial](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmarks and Performance

*   See the release blogs for detailed performance information:
    *   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
    *   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
    *   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
    *   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is a production-ready LLM inference engine deployed at scale, generating trillions of tokens daily and trusted by leading organizations.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us

For enterprise adoption, technical consulting, sponsorship, or partnership inquiries, contact us at contact@sglang.ai.

## Acknowledgment

SGLang builds upon and acknowledges the contributions of the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).