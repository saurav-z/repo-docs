<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
</div>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

---

## SGLang: Unleash the Power of LLMs with High-Performance Serving

SGLang is a blazing-fast serving framework for large language models (LLMs) and vision language models, engineered for speed and control.  [**Visit the original repository on GitHub**](https://github.com/sgl-project/sglang).

### Key Features:

*   **Blazing-Fast Backend Runtime:** Experience unparalleled efficiency with RadixAttention, zero-overhead CPU scheduling, prefill-decode disaggregation, speculative decoding, and more.  SGLang supports a wide range of optimization techniques like continuous batching, paged attention, tensor parallelism, pipeline parallelism, expert parallelism, structured outputs, chunked prefill, quantization (FP8/INT4/AWQ/GPTQ), and multi-lora batching.
*   **Intuitive Frontend Language:** Easily program LLM applications with a flexible interface, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
*   **Broad Model Compatibility:** Seamlessly integrate with leading generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), and easily extend to new models.
*   **Active Community & Industry Adoption:** Benefit from a vibrant open-source community and industry support, with SGLang powering trillions of tokens daily.

### News

*   [2025/06] 🔥 Awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
*   [2025/06] 🔥 Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   [2025/05] 🔥 Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   [2025/01] 🔥 SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   [2024/12] 🔥 v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

*   [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

### Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start Tutorial](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Performance & Benchmarks

*   Explore the latest performance gains in the following release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).

### Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

### Adoption and Sponsorship

SGLang is the trusted LLM inference engine deployed at scale by industry leaders, generating trillions of tokens daily. Key adopters include:

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

### Contact Us

For enterprise inquiries, including technical consulting, sponsorship, and partnerships, reach out to us at contact@sglang.ai.

### Acknowledgment

SGLang draws inspiration and reuses code from: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).