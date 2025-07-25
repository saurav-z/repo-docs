<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

---

## SGLang: High-Performance LLM Serving for Trillions of Tokens

**SGLang** is a cutting-edge serving framework designed to accelerate large language model (LLM) and vision language model (VLM) inference, optimizing performance and control. [Explore the original repository on GitHub](https://github.com/sgl-project/sglang).

**Key Features:**

*   **Blazing Fast Backend Runtime:**
    *   RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation.
    *   Speculative decoding.
    *   Continuous batching and Paged Attention.
    *   Tensor, pipeline, and expert parallelism support.
    *   Structured outputs and chunked prefill.
    *   Quantization: FP8/INT4/AWQ/GPTQ.
    *   Multi-LoRA batching.

*   **Flexible Frontend Language:**
    *   Intuitive interface for LLM application development.
    *   Chained generation calls.
    *   Advanced prompting capabilities.
    *   Control flow integration.
    *   Multi-modal input support.
    *   Parallel processing.
    *   External interactions.

*   **Extensive Model Support:**
    *   Wide range of generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.
    *   Embedding models: e5-mistral, gte, mcdse.
    *   Reward models: Skywork.
    *   Easy extensibility for new model integration.

*   **Active Community and Industry Adoption:**
    *   Open-source project with an active community.
    *   Trusted by industry leaders and deployed at scale.

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmarks and Performance

Learn more about SGLang's performance advantages in these blog posts:
*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is a production-ready LLM inference engine deployed at scale, generating trillions of tokens daily. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia. As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 1,000,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact

For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgements

We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

---

**Additional Resources:**

*   [Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
*   [Documentation](https://docs.sglang.ai/)
*   [Join Slack](https://slack.sglang.ai/)
*   [Join Bi-Weekly Development Meeting](https://meeting.sglang.ai/)
*   [Slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides)
```
Key improvements and explanations:

*   **SEO Optimization:**  The title now includes the main keyword "SGLang" and the core benefit "High-Performance LLM Serving".  The descriptions also include the core keywords.
*   **Clear Structure:**  Uses headings (H2) for organization, making the content easy to scan.
*   **Concise Summary:** The one-sentence hook immediately grabs attention.
*   **Bulleted Lists:**  Key features are presented using bullet points, enhancing readability and making it easy to identify benefits.
*   **Keyword Integration:**  Keywords are naturally integrated throughout the text (e.g., "LLM serving," "large language models," "high-performance," "inference").
*   **Call to Action:** Encourages exploration of the original repository with a clear link.
*   **Focus on Benefits:** Highlights the advantages of using SGLang.
*   **Removed Redundancy:**  Eliminated repetitive phrases.
*   **Concise descriptions:** Made descriptions of features as concise as possible.
*   **Emphasis on industry adoption:** Added this as its own section.
*   **Cleaned up code:** Removed redundant HTML and extraneous spacing.