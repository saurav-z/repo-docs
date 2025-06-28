<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>
  <a href="https://github.com/sgl-project/sglang">
    <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/sgl-project/sglang?style=social">
  </a>
  <br>
  [![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)](https://pypi.org/project/sglang)
  [![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
  [![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)
</div>

---

**SGLang is a high-performance serving framework that supercharges your large language models (LLMs) and vision language models (VLMs), making them faster and more controllable.**

## Key Features

*   **Blazing Fast Backend Runtime:** Experience efficient serving with cutting-edge features:
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
*   **Intuitive Frontend Language:** Simplify LLM application development with an easy-to-use interface:
    *   Chained generation calls.
    *   Advanced prompting capabilities.
    *   Control flow structures.
    *   Multi-modal input handling.
    *   Parallel processing.
    *   External interaction capabilities.
*   **Extensive Model Support:** SGLang seamlessly integrates with a wide array of models:
    *   Generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, and more.
    *   Embedding models: e5-mistral, gte, mcdse.
    *   Reward models: Skywork.
    *   Easy extensibility for adding new models.
*   **Active Community and Industry Adoption:** Benefit from a vibrant open-source community and real-world deployments.

## What's New

*   **[2025/05]** Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   **[2025/03]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   **[2025/03]** SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   **[2025/01]** SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   **[2024/12]** v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   **[2024/07]** v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
*   **[More News](https://github.com/sgl-project/sglang#news)**

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance and Benchmarks

See the following blog posts for detailed performance analysis:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is a trusted and widely adopted solution, powering large-scale deployments across various industries.  Deployed at scale and generating trillions of tokens in production daily, SGLang is trusted and adopted by leading enterprises and institutions, including xAI, NVIDIA, AMD, Google Cloud, Oracle Cloud, LinkedIn, Cursor, Voltage Park, Atlas Cloud, DataCrunch, Baseten, Nebius, Novita, InnoMatrix, RunPod, Stanford, UC Berkeley, UCLA, ETCHED, Jam & Tea Studios, Hyperbolic, and major technology organizations.  As an open-source LLM inference engine, SGLang has become the de facto standard in the industry, with production deployments running on over 100,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact Us

For inquiries regarding large-scale deployments, technical consulting, sponsorships, or partnerships, please contact us at contact@sglang.ai.

## Resources

*   [Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
*   [Documentation](https://docs.sglang.ai/)
*   [Join Slack](https://slack.sglang.ai/)
*   [Join Bi-Weekly Development Meeting](https://meeting.sglang.ai/)
*   [Slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides)

## Acknowledgment

SGLang leverages and acknowledges the work of these projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

[**View the original repository on GitHub**](https://github.com/sgl-project/sglang)
```
Key improvements and SEO considerations:

*   **Strong Title and Hook:**  The opening sentence is a clear, concise value proposition.
*   **Keyword Optimization:** Includes terms like "large language models (LLMs)", "vision language models (VLMs)", "high-performance", and core features to help with search.
*   **Clear Headings:**  Uses headings to structure content for readability and SEO.
*   **Bulleted Key Features:** Makes it easy for users to scan and understand the core benefits.
*   **Specific Value Propositions:**  Highlights the advantages of SGLang (speed, control, model support).
*   **Links to Key Resources:**  Provides easy access to documentation, tutorials, and the original repo.
*   **Call to Action:** Encourages contact for enterprise adoption.
*   **Concise and Direct:** Avoids unnecessary jargon.
*   **Focus on Benefits:**  Emphasizes what users gain from using SGLang.
*   **Clear Updates Section:** The "What's New" section includes the most recent updates.
*   **Social Proof:**  Mentions the widespread adoption and the impressive list of companies using SGLang.
*   **Acknowledgement:** Clearly acknowledges and credits the projects SGLang is based on.
*   **"View the original repository on GitHub":** A call-to-action link to the original repo.