<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <h1>SGLang: The Fastest Way to Serve LLMs</h1>
  
  [![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
  ![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
  [![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
  [![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)
</div>

**SGLang is a high-performance serving framework that accelerates large language model (LLM) inference, used by leading enterprises to serve trillions of tokens daily.** ([Original Repo](https://github.com/sgl-project/sglang))

## Key Features

*   **Blazing-Fast Backend Runtime:**
    *   RadixAttention for efficient prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation for improved performance.
    *   Speculative decoding and continuous batching.
    *   Paged attention and tensor/pipeline/expert parallelism.
    *   Support for structured outputs and chunked prefill.
    *   Quantization (FP8/INT4/AWQ/GPTQ) and multi-LoRA batching.
*   **Flexible Frontend Language:**
    *   Intuitive interface for LLM application development.
    *   Chained generation calls, advanced prompting, and control flow.
    *   Multi-modal inputs and parallel processing capabilities.
    *   Seamless integration with external interactions.
*   **Extensive Model Support:**
    *   Wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA).
    *   Support for various embedding models (e5-mistral, gte, mcdse) and reward models (Skywork).
    *   Easy extensibility for incorporating new models.
*   **Thriving Community & Industry Adoption:** SGLang is open-source and backed by an active community, with industry-wide adoption.

## News & Updates

*   **[June 2025]** Awarded Open Source AI Grant by a16z.
*   **[June 2025]** Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP: 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   **[May 2025]** Deploying DeepSeek with PD and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   **[March 2025]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)).
*   **[March 2025]** SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/)).
*   **[December 2024]** v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   **[July 2024]** v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More News</summary>

*   **[February 2025]** Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   **[January 2025]** SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   **[October 2024]** The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   **[September 2024]** v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   **[February 2024]** SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   **[January 2024]** SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   **[January 2024]** SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).
</details>

## Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance & Benchmarks

Explore SGLang's performance advantages in the following blog posts:
*   [v0.2 Release](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 Release](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 Release](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption

<p>SGLang is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia.  As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 1,000,000 GPUs worldwide.</p>
<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="SGLang Adoption" width="800" margin="10px">

## Contact & Support

For inquiries regarding enterprise adoption, technical consulting, sponsorship, or partnerships, please contact us at: contact@sglang.ai.

## Acknowledgements

SGLang builds upon and acknowledges the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).