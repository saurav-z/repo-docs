<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <h1>SGLang: High-Performance LLM Serving Framework</h1>

  [![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)]()
  [![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
  [![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)
</div>

SGLang is a cutting-edge serving framework designed to accelerate and optimize Large Language Models (LLMs), used to power trillions of tokens daily. **[Explore the SGLang repository on GitHub](https://github.com/sgl-project/sglang).**

**Key Features:**

*   🚀 **High-Performance Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation & Speculative decoding
    *   Continuous batching & Paged attention
    *   Tensor/Pipeline/Expert Parallelism
    *   Structured outputs & Chunked prefill
    *   Quantization support (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching
*   💻 **Flexible Frontend Language:**
    *   Intuitive interface for LLM application development
    *   Chained generation calls & Advanced prompting
    *   Control flow mechanisms
    *   Multi-modal inputs
    *   Parallel processing capabilities
    *   External interaction support
*   🌐 **Extensive Model Support:**
    *   Supports a wide array of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.)
    *   Compatibility with embedding models (e5-mistral, gte, mcdse) & reward models (Skywork)
    *   Easy extensibility for integrating new models
*   🤝 **Active Community:**
    *   Open-source with a vibrant and supportive community.

## News and Updates

*   [2025/06] 🔥 Awarded a16z Open Source AI Grant ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
*   [2025/06] 🔥 Deploying DeepSeek on GB200 NVL72 ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   [2025/05] 🔥 Deploying DeepSeek with PD and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   [2025/03] SGLang Joins PyTorch Ecosystem ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
<details>
<summary>More</summary>

*   [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   [2024/02] SGLang enables **3x faster JSON decoding** ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>
## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmark and Performance

Explore performance gains in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

## Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship

SGLang is a trusted solution deployed at scale by leading organizations:

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">

## Contact Us

For enterprise inquiries, including technical consulting, sponsorship opportunities, and partnerships, please reach out to us at contact@sglang.ai.

## Acknowledgment

We would like to acknowledge the following projects, from which we learned and reused code: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).