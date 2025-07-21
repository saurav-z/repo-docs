<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang Logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

---

## SGLang: The High-Performance LLM Serving Framework

**SGLang is a cutting-edge serving framework designed to accelerate and streamline the deployment of Large Language Models (LLMs) and Vision Language Models (VLMs).**

[View the original repository on GitHub](https://github.com/sgl-project/sglang)

### Key Features

*   **Blazing-Fast Backend Runtime:**
    *   Efficient serving with RadixAttention for prefix caching.
    *   Zero-overhead CPU scheduler.
    *   Prefill-decode disaggregation for optimized processing.
    *   Speculative decoding to improve generation speed.
    *   Continuous batching for handling varying request loads.
    *   Paged attention for memory efficiency.
    *   Tensor, pipeline, and expert parallelism for scaling.
    *   Support for structured outputs.
    *   Chunked prefill for faster initial processing.
    *   Quantization support (FP8/INT4/AWQ/GPTQ).
    *   Multi-LoRA batching for flexibility.
*   **Intuitive Frontend Language:**
    *   Chained generation calls for complex tasks.
    *   Advanced prompting capabilities.
    *   Control flow structures for conditional logic.
    *   Multi-modal input support for diverse data types.
    *   Parallelism for concurrent operations.
    *   External interactions for integrating with other systems.
*   **Extensive Model Compatibility:**
    *   Supports a wide array of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, and more).
    *   Compatible with various embedding models (e5-mistral, gte, mcdse).
    *   Includes support for reward models (Skywork).
    *   Easy extensibility for adding new models.
*   **Active Community & Industry Adoption:** Open-source with a thriving community and widespread industry adoption.

### Quick Links

*   [**Documentation**](https://docs.sglang.ai/)
*   [**Getting Started**](https://docs.sglang.ai/start/install.html)
*   [**Quick Start Tutorial**](https://docs.sglang.ai/backend/send_request.html)
*   [**Community Slack**](https://slack.sglang.ai/)
*   [**Roadmap**](https://github.com/sgl-project/sglang/issues/4042)
*   [**Blog**](https://lmsys.org/blog/2025-05-05-large-scale-ep/)

### News & Updates

*   **(June 2025)** SGLang receives the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
*   **(June 2025)** Deployment of DeepSeek on GB200 NVL72 with PD and Large Scale EP, achieving 2.7x higher decoding throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
*   **(May 2025)** DeepSeek deployment with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
*   **(March 2025)** Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)).
*   **(March 2025)** SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/)).
*   **(December 2024)** v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
*   **(July 2024)** v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<!-- You can add more news items here, expanding the details as needed. -->

### Benchmark & Performance

*   Explore the performance gains in these release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

### Adoption and Sponsorship

SGLang is powering trillions of tokens daily in production and is used by numerous leading organizations:
[<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px">](https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png)

(Including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia.)

### Contact Us

For enterprise inquiries regarding large-scale deployments, technical consulting, sponsorship opportunities, or partnerships, please contact us at [contact@sglang.ai](mailto:contact@sglang.ai).

### Acknowledgment

We thank the following projects for their influence: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
```
Key improvements and SEO considerations:

*   **Clear, concise title and one-sentence hook:**  "SGLang: The High-Performance LLM Serving Framework" and the opening sentence.
*   **Keywords:**  Included throughout (LLM, VLM, serving framework, high-performance, etc.).
*   **Headings:** Properly formatted for structure and SEO.
*   **Bulleted Key Features:** Easy to read and highlights the main selling points.
*   **Concise Descriptions:** Descriptions of features are short and focused.
*   **Quick Links:**  Important links are easily accessible.
*   **News/Updates Section:**  Keeps the content fresh and shows activity.  Added more context to the news items.
*   **Adoption/Sponsorship Section:** Highlights adoption and sponsorship, important for credibility.  Kept the image.
*   **Contact Information:**  Provided a clear call to action.
*   **Acknowledgment Section:**  Gives credit where it's due.
*   **Removed unnecessary HTML:**  Cleaned up the code for better readability.
*   **SEO-Friendly Formatting:** Bolded key terms and used headings effectively.
*   **Internal and External Links:** Properly linked to documentation, blogs, and other relevant resources, including linking back to the original repo.
*   **ALT Text:** Added alt text for the logo image.