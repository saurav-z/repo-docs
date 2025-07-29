<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

## SGLang: The High-Performance LLM Serving Framework

**SGLang is a blazing-fast serving framework that empowers developers to build and deploy large language models with unparalleled speed and efficiency.**  [Explore the original repository on GitHub](https://github.com/sgl-project/sglang).

### Key Features

*   **Blazing-Fast Backend Runtime:** Optimized for speed with features like RadixAttention, zero-overhead CPU scheduling, and support for advanced techniques such as prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor parallelism, expert parallelism, structured outputs, quantization (FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
*   **Flexible Frontend Language:**  Provides an intuitive interface for building LLM applications, supporting chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external integrations.
*   **Broad Model Support:** Works with a wide range of popular generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, and more), embedding models, and reward models, with easy extensibility for integrating new models.
*   **Active Community & Industry Adoption:**  Open-source and widely adopted, trusted by leading enterprises and institutions, generating trillions of tokens daily.

### Getting Started

*   [Install SGLang](https://docs.sglang.ai/start/install.html)
*   [Quick Start](https://docs.sglang.ai/backend/send_request.html)
*   [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
*   [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
*   [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

### Benchmarks and Performance

For detailed performance analysis, explore the following blog posts:

*   [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)
*   [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)
*   [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)
*   [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

### News

*   [2025/06] 🔥 SGLang awarded Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
*   [2025/06] 🔥 Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   [2025/05] 🔥 Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
*   [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
*   [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

*   [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
*   [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
*   [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
*   [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
*   [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
*   [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
*   [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

### Roadmap

*   [Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

### Adoption and Sponsorship

SGLang is a trusted solution deployed at scale, generating trillions of tokens daily across a wide range of industries.  It is utilized by major organizations including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Adoption" width="800" margin="10px">

### Contact

For enterprise inquiries, including technical consulting, sponsorship, or partnership opportunities, please reach out to us at contact@sglang.ai.

### Acknowledgment

We would like to acknowledge the following projects for their influence on our design and code: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
```
Key improvements and explanations:

*   **SEO Optimization:** The title includes "LLM Serving Framework," "High-Performance," and uses keywords like "large language models" to improve search visibility.
*   **Concise Hook:** The one-sentence hook immediately grabs the reader's attention and states the main benefit.
*   **Clear Headings:**  The document is well-structured with clear headings (Key Features, Getting Started, Benchmarks, News, Roadmap, Adoption, Contact, Acknowledgment) for readability and scannability.
*   **Bulleted Key Features:** This makes it easy for users to quickly understand the core value proposition.
*   **Link Back to Original Repo:** Includes a direct link to the GitHub repository for easy access.
*   **Actionable Information:** Includes links to installation, quick start, and tutorials.
*   **Focus on Benefits:** The descriptions of features highlight the benefits (e.g., "Blazing-Fast Backend Runtime: Optimized for speed...")
*   **Stronger Language:** Uses more active and engaging language throughout.
*   **Visual Aid:** Kept the adoption image to showcase real-world usage.
*   **Contact Information:** Includes contact information for enterprise users.
*   **Summarization:** The original text was condensed to highlight the most important information.
*   **Removed Unnecessary Markup:** Cleaned up some extra HTML markup.
*   **Clear Benchmarks Section:** Made the Benchmarks and Performance section more concise and user-friendly.
*   **Alt Text for Images:** Added `alt` text for images to improve accessibility and SEO.
*   **Consistent Formatting:** Maintained consistent formatting throughout the document.