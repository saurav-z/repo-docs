<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <h1>SGLang: High-Performance LLM Serving for Production</h1>
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

<br>

**SGLang is a cutting-edge, open-source serving framework designed to supercharge the performance and control of large language models (LLMs) and vision language models (VLMs).**

<br>

## Key Features:

*   🚀 **Blazing-Fast Backend Runtime:** Experience efficient serving with RadixAttention, zero-overhead CPU scheduling, and advanced features like prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert parallelism, structured outputs, and more.
*   💻 **Intuitive Frontend Language:** Easily program LLM applications with a flexible interface, enabling chained generation, advanced prompting, control flow, multi-modal input, and seamless external interactions.
*   🧠 **Extensive Model Support:** Leverage SGLang's broad compatibility with a wide array of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse), and reward models (Skywork), with easy extensibility.
*   🤝 **Active Community & Industry Adoption:** Benefit from a vibrant open-source community and widespread industry adoption, with deployments generating trillions of tokens daily, and trusted by leading companies and institutions.
*   💰 **Optimized for Efficiency:** Supports quantization (FP8/INT4/AWQ/GPTQ) and multi-LoRA batching to maximize performance while minimizing resource consumption.

<br>

## Getting Started

*   [Installation](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

<br>

## News

*   **[2025/06]** SGLang receives an a16z Open Source AI Grant.
*   **[2025/06]** GB200 NVL72 deployment with 2.7x higher decoding throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   **[2025/05]** Deploying DeepSeek with PD and large-scale expert parallelism ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   **[2025/03]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   **[2025/03]** SGLang Joins PyTorch Ecosystem ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   **[2024/12]** v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   **[2024/07]** v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

*   **[2025/02]** Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   **[2025/01]** SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   **[2024/10]** The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   **[2024/09]** v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   **[2024/02]** SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   **[2024/01]** SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   **[2024/01]** SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

<br>

## Benchmark and Performance

Explore detailed performance insights in the release blogs: [v0.2](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), and [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

<br>

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

<br>

## Adoption and Sponsorship

SGLang is a production-ready solution trusted by leading organizations across various industries.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption by Industry Leaders" width="800" margin="10px">

<br>

## Contact Us

For enterprise inquiries, including technical consulting, sponsorships, or partnership opportunities, please reach out to us at [contact@sglang.ai](mailto:contact@sglang.ai).

<br>

## Acknowledgment

SGLang incorporates design and code from these projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

<br>

---

[**View the original SGLang repository on GitHub**](https://github.com/sgl-project/sglang)