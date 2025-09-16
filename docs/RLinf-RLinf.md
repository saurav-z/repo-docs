<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
  <a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
  <a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
  <a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
  <a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpeg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<h1 align="center">
  RLinf: Revolutionizing Reinforcement Learning for Agentic AI
</h1>

RLinf is a cutting-edge, open-source infrastructure designed to supercharge the training of foundation models through reinforcement learning, enabling unparalleled scalability and flexibility.  

**[Explore the RLinf Repository on GitHub](https://github.com/RLinf/RLinf)**

## Key Features

*   **Macro-to-Micro Flow (M2Flow):** A novel paradigm that decouples logical workflow construction from physical execution, enhancing programmability and efficiency.
*   **Flexible Execution Modes:** Supports collocated, disaggregated, and hybrid modes for optimal resource utilization.
*   **Intelligent Auto-Scheduling:** Automatically selects the best execution mode based on the training workload, eliminating manual configuration.
*   **Embodied Agent Support:**  Seamlessly integrates with leading VLA models (OpenVLA, OpenVLA-OFT, π₀) and simulators (ManiSkill3, LIBERO).
*   **Unrivaled Speed & Efficiency:** Achieves a **120%+** throughput improvement with fine-grained pipelining in Hybrid mode and dynamic online scaling.
*   **Multiple Backend Integrations:** Provides rapid adaptation using FSDP + Hugging Face and maximized efficiency through Megatron + SGLang.
*   **Built-in RL Methods:** Supports popular algorithms like PPO, GRPO, DAPO, Reinforce++, and more.

## What's New

*   **[2025/08] RLinf is open-sourced!** The official v0.1 release is coming soon! The paper [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation]() will also be released.

## Roadmap

### System-Level Enhancements
*   [ ] Support for heterogeneous GPUs
*   [ ] Support for asynchronous pipeline execution
*   [ ] Support for Mixture of Experts (MoE)
*   [ ] Support for vLLM inference backend

### Application-Level Extensions
*   [ ] Support for Vision-Language Models (VLMs) training
*   [ ] Support for deep searcher agent training
*   [ ] Support for multi-agent training
*   [ ] Support for integration with more embodied simulators (e.g., [Meta-World](https://github.com/Farama-Foundation/Metaworld), [GENESIS](https://github.com/Genesis-Embodied-AI/Genesis))
*   [ ] Support for more Vision Language Action models (VLAs), such as [GR00T](https://github.com/NVIDIA/Isaac-GR00T)
*   [ ] Support for world model
*   [ ] Support for real-world RL embodied intelligence

## Getting Started

Comprehensive documentation is available [**Here**](https://rlinf.readthedocs.io/en/latest/).

**Quickstart Guides:**

*   [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
*   [Quickstart 1: PPO Training of VLAs on Maniskill3](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
*   [Quickstart 2: GRPO Training of LLMs on MATH](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
*   [Multi-node Training](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
*   [Model Evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)

**Key Design:**

*   [Unified User Interface Usage](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/index.html)
*   [Flexible Execution Modes](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html)
*   [Enable Automatic Scheduling](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/scheduler/index.html)
*   [Elastic Communication](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/communication/index.html)

**Example Gallery:**

*   [Embodied Intelligence Vision-Language-Action Model training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied.html)
*   [Math Reasoning Model Training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html)

**Advanced Features:**

*   [5D Parallelism Configuration for Megatron-LM](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html)
*   [LoRA Integration for efficient fine-tuning](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/lora.html)
*   [Switch between different versions of SGLang](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/version.html)
*   [Checkpoint Resume and Recovery Support](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html)

**Extending The Framework:**

*   [Adding new Environments](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html)
*   [Adding new Models with FSDP+Huggingface backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_fsdp.html)
*   [Adding new Models with Megatron+SGLang backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_megatron.html)

**Blogs:**

*   [Comparison with VeRL](https://rlinf.readthedocs.io/en/latest/rst_source/blog/compare_with_verl.html)

## Build Status

| Type               | Status                                                                                                                                 |
| :----------------: | :------------------------------------------------------------------------------------------------------------------------------------- |
| Reasoning RL-MATH  | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA    | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions! Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before contributing.

## Citation and Acknowledgements

If you find **RLinf** helpful, please cite the GitHub repository:

```bibtex
@misc{RLinf_repo,
  title        = {RLinf: Reinforcement Learning Infrastructure for Agentic AI},
  howpublished = {\url{https://github.com/RLinf/RLinf}},
  note         = {GitHub repository},
  year         = {2025}
}
```

**Paper**: A full paper describing RLinf will be released by **September 20, 2025**. We will update this section with the official citation and BibTeX when they become available.

**Acknowledgements**

RLinf is inspired by and benefits from the broader open-source community. We would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP). If your project or contribution is inadvertently missed, please open an issue or pull request.

**Contact:**

We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!

-   Chao Yu: zoeyuchao@gmail.com
-   Yu Wang: yu-wang@tsinghua.edu.cn
```
Key improvements and SEO optimizations:

*   **Compelling Hook:** "RLinf: Revolutionizing Reinforcement Learning for Agentic AI" immediately grabs attention.
*   **SEO Keywords:** Incorporated relevant keywords like "Reinforcement Learning," "Agentic AI," "Foundation Models," and related terms throughout the headings and text.
*   **Clear Headings:**  Organized the README with clear, descriptive headings (Key Features, What's New, Roadmap, etc.).
*   **Bulleted Lists:** Used bullet points to highlight key features, benefits, and quickstart guides.
*   **Concise Language:**  Phrasing is more direct and easier to read.
*   **Call to Action:** Encourages users to explore the repository.
*   **Enhanced Formatting:** Used bolding for emphasis and better readability.
*   **Paper Release Highlight:**  The paper release date is emphasized.
*   **Contact Info Prominence:** Contact information remains easily accessible.