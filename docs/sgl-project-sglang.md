<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" alt="GitHub Stars">
  </a>
</div>

---

# SGLang: High-Performance LLM and Vision Language Model Serving

**SGLang is a cutting-edge, open-source serving framework designed to accelerate and optimize the deployment of large language models (LLMs) and vision language models (VLMs).**

## Key Features

*   **Fast Backend Runtime:** SGLang's backend delivers exceptional performance through:
    *   RadixAttention for efficient prefix caching
    *   Zero-overhead CPU scheduler
    *   Prefill-decode disaggregation
    *   Speculative decoding
    *   Continuous batching
    *   Paged attention
    *   Tensor, pipeline, and expert parallelism
    *   Structured outputs
    *   Chunked prefill
    *   Quantization support (FP8/INT4/AWQ/GPTQ)
    *   Multi-LoRA batching
*   **Flexible Frontend Language:** Simplifies LLM application development with:
    *   Chained generation calls
    *   Advanced prompting capabilities
    *   Control flow mechanisms
    *   Multi-modal input support
    *   Parallel processing
    *   External interaction capabilities
*   **Extensive Model Support:** Compatible with a wide range of models:
    *   Generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, and more)
    *   Embedding models (e5-mistral, gte, mcdse)
    *   Reward models (Skywork)
    *   Easy extensibility for new model integrations
*   **Active Community and Industry Adoption:** Open-source with strong community support, powering large-scale deployments across leading organizations and generating trillions of tokens daily.

## News and Updates

*   **[2025/06]** SGLang receives an Open Source AI Grant from a16z.
*   **[2025/06]** GB200 NVL72 with PD and Large Scale EP: 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   **[2025/05]** Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   **[2025/03]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   **[2025/03]** SGLang Joins PyTorch Ecosystem ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   **[2025/01]** Day one support for DeepSeek V3/R1 models with DeepSeek-specific optimizations.

<details>
<summary>More News</summary>

*   **[2025/02]** Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   **[2024/10]** The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sglang/issues/4042))
*   **[2024/09]** v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   **[2024/02]** 3x faster JSON decoding with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   **[2024/01]** Up to 5x faster inference with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   **[2024/01]** Powers serving of the official LLaVA v1.6 demo.
</details>

## Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Benchmarks and Performance

Explore detailed performance insights in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/).

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is the trusted serving engine for LLMs and VLMs used by a wide range of leading organizations.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

## Contact Us

For inquiries regarding adoption, large-scale deployment, technical consulting, sponsorship opportunities, or partnerships, please reach out to us at [contact@sglang.ai](mailto:contact@sglang.ai).

## Acknowledgment

We gratefully acknowledge the influence and code contributions from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).

---

**[Explore the SGLang project on GitHub](https://github.com/sgl-project/sglang)**