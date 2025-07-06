<div align="center">
  <img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px">
  <br>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" alt="GitHub stars">
  </a>
</div>

---

# SGLang: High-Performance LLM Serving Framework

**SGLang is a cutting-edge serving framework designed to accelerate and enhance your Large Language Model (LLM) and Vision Language Model (VLM) interactions.** [Explore the SGLang Repository](https://github.com/sgl-project/sglang)

**Key Features:**

*   **Blazing-Fast Backend Runtime:** SGLang's runtime optimizes LLM serving through a suite of techniques including RadixAttention, zero-overhead CPU scheduling, and expert parallelism to deliver maximum throughput and low latency.
*   **Intuitive Frontend Language:** Easily program LLM applications with a flexible and intuitive frontend language, supporting chained generation calls, advanced prompting, control flow, multi-modal inputs, and external interactions.
*   **Extensive Model Compatibility:** Broad support for various LLMs (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models, and reward models, with simple integration for new models.
*   **Active and Supportive Community:** Benefit from an open-source community and industry adoption from leading enterprises, with extensive documentation and support.

## News

*   **[2025/06]** SGLang awarded Open Source AI Grant by a16z.
*   **[2025/06]** 2.7x Higher Decoding Throughput on GB200 NVL72 with PD and Large Scale EP ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   **[2025/05]** Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   **[2025/03]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   **[2025/03]** SGLang Joins PyTorch Ecosystem ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   **[2025/01]** SGLang provides day one support for DeepSeek V3/R1 models ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   **[2024/12]** v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   **[2024/07]** v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More News</summary>

*   **[2025/02]** Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   **[2024/10]** The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sglang-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   **[2024/09]** v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   **[2024/02]** SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   **[2024/01]** SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   **[2024/01]** SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

## Performance & Benchmarks

See the release blogs for detailed performance comparisons:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)

## Roadmap

*   [Development Roadmap (2025 H1)](https://github.com/sgl-project/sglang/issues/4042)

## Adoption and Sponsorship

SGLang is a production-ready LLM inference engine generating trillions of tokens daily. It is trusted and adopted by leading organizations like xAI, NVIDIA, AMD, Google Cloud, and more.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

## Contact

For business inquiries regarding adoption, deployment, technical consulting, sponsorship, or partnerships, please contact us at [contact@sglang.ai](mailto:contact@sglang.ai).

## Acknowledgments

SGLang incorporates design elements and code from [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).