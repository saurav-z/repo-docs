<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)
</div>

## SGLang: Supercharge Your LLM Performance

**SGLang** is a high-performance serving framework designed to accelerate Large Language Model (LLM) and Vision Language Model (VLM) deployments. [View the original repository on GitHub](https://github.com/sgl-project/sglang).

### Key Features:

*   🚀 **High-Performance Backend Runtime:** Experience efficient serving with optimized features like RadixAttention, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, and more.
*   💻 **Flexible Frontend Language:** Program LLM applications intuitively with chained generation calls, advanced prompting, control flow, multi-modal inputs, and external interactions.
*   🌐 **Extensive Model Support:** Supports a wide array of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility.
*   🤝 **Active Community & Industry Adoption:** Benefit from an open-source project with a thriving community and wide-scale adoption by leading enterprises and research institutions.

### Why Choose SGLang?

*   **Speed:** Achieve faster inference speeds compared to traditional LLM serving solutions.
*   **Efficiency:** Optimize resource utilization with features like batch scheduling and cache-aware load balancing.
*   **Control:** Gain more control over your LLM applications with a flexible frontend language.

### Key Benefits:

*   **Reduced Latency:** Significantly decrease the time it takes to generate responses.
*   **Increased Throughput:** Handle more requests concurrently.
*   **Lower Costs:** Optimize hardware utilization, leading to reduced infrastructure expenses.

### News & Updates

*   **[2025/06]** Awarded Open Source AI Grant by a16z.
*   **[2025/06]** Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP: 2.7x Higher Decoding Throughput.
*   **[2025/05]** Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs.
*   **[2025/03]** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X.
*   **[2025/03]** SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine.
*   **[2024/12]** v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs.
*   **[2024/07]** v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM).

<details>
<summary>More News</summary>

*   **[2025/02]** Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU
*   **[2025/01]** SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations.
*   **[2024/10]** The First SGLang Online Meetup.
*   **[2024/09]** v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision.
*   **[2024/02]** SGLang enables **3x faster JSON decoding** with compressed finite state machine.
*   **[2024/01]** SGLang provides up to **5x faster inference** with RadixAttention.
*   **[2024/01]** SGLang powers the serving of the official **LLaVA v1.6** release demo.

</details>

### Getting Started

*   [Installation Guide](https://docs.sglang.ai/start/install.html)
*   [Quick Start Guide](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Benchmarks & Performance

Explore performance improvements in the following blog posts:
*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

### Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

### Adoption & Sponsorship

SGLang powers trillions of tokens daily, trusted by industry leaders including:

*   xAI
*   AMD
*   NVIDIA
*   Intel
*   LinkedIn
*   And many more...

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="SGLang Adoption" width="800" margin="10px">

### Contact Us

For inquiries regarding enterprise adoption, technical consulting, sponsorships, or partnerships, please contact us at: contact@sglang.ai

### Acknowledgments

SGLang's design and code were inspired by: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).