<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>

  [![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
  [![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)]
  [![License](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
  [![Issue Resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Open Issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)
</div>

---

**SGLang: Accelerate Your LLM Inference with High-Performance Serving.** [Check out the original repo](https://github.com/sgl-project/sglang)

## Key Features

*   **Blazing-Fast Backend Runtime:** Experience efficient LLM serving with cutting-edge features.
    *   RadixAttention for efficient prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation for optimized processing.
    *   Speculative decoding for faster generation.
    *   Continuous batching for improved throughput.
    *   Paged attention for memory efficiency.
    *   Tensor, pipeline, and expert parallelism for scaling.
    *   Support for structured outputs.
    *   Chunked prefill for large inputs.
    *   Quantization support (FP8/INT4/AWQ/GPTQ) for reduced memory footprint.
    *   Multi-LoRA batching for flexible model management.

*   **Intuitive Frontend Language:** Simplify LLM application development with a user-friendly interface.
    *   Chained generation calls for complex workflows.
    *   Advanced prompting capabilities.
    *   Control flow for dynamic logic.
    *   Multi-modal input support.
    *   Parallelism for concurrent operations.
    *   External interaction capabilities.

*   **Extensive Model Compatibility:** Supports a wide array of models out-of-the-box.
    *   Generative models: Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, and more.
    *   Embedding models: e5-mistral, gte, mcdse.
    *   Reward models: Skywork.
    *   Easy extensibility for integrating new models.

*   **Vibrant Community and Industry Adoption:** Benefit from an active open-source community.
    *   Trusted by industry leaders and deployed at scale.

## News

*   [2025/06] SGLang awarded an Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
*   [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   [2025/01] Day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
*   [More Recent Updates](https://github.com/sgl-project/sglang#news)

<details>
<summary>Older News</summary>

*   [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sglang-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start Tutorial](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmark and Performance

For detailed performance benchmarks and comparisons, refer to the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is a production-ready LLM inference engine adopted by industry leaders and deployed at scale, handling trillions of tokens daily. Notable users include xAI, NVIDIA, AMD, Google Cloud, Oracle Cloud, LinkedIn, Cursor, Voltage Park, Atlas Cloud, DataCrunch, Baseten, Nebius, Novita, InnoMatrix, RunPod, Stanford, UC Berkeley, UCLA, ETCHED, Jam & Tea Studios, Hyperbolic, and major tech organizations worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

## Contact Us

For inquiries about enterprise adoption, technical consulting, sponsorship opportunities, or partnership, please contact us at contact@sglang.ai.

## Acknowledgments

SGLang builds upon the knowledge and code of these projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).