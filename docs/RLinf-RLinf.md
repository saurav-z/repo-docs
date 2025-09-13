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

# RLinf: The Cutting-Edge Reinforcement Learning Infrastructure for Agentic AI

**RLinf empowers researchers and developers to build and scale agentic AI with a flexible and efficient open-source infrastructure.**  [Explore the RLinf GitHub Repository](https://github.com/RLinf/RLinf).

## Key Features

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Decouples logical workflow construction from physical communication and scheduling for enhanced flexibility and efficiency.
*   **Flexible Execution Modes:**  Supports collocated, disaggregated, and hybrid modes for optimal resource utilization and performance.
*   **Automated Scheduling:** Dynamically selects the best execution mode based on the training workload, eliminating manual resource allocation.
*   **Embodied Agent Support:**  Seamlessly integrates with popular VLA models (OpenVLA, OpenVLA-OFT, π₀) and simulators (ManiSkill3, LIBERO) for advanced agent training.
*   **High-Performance Training:**  Achieves over 120% throughput improvement with hybrid mode and fine-grained pipelining, and boosts efficiency by 20-40% via automatic online scaling.
*   **Multiple Backend Integrations:**  Offers versatility with FSDP + Hugging Face for rapid prototyping and Megatron + SGLang for large-scale, high-efficiency training.
*   **Built-in RL Methods:**  Includes out-of-the-box support for PPO, GRPO, DAPO, Reinforce++, and other popular reinforcement learning algorithms.

## What's New

*   **[2025/08] Open-Sourced and Early Access:** RLinf is now publicly available. The formal v0.1 release and accompanying research paper are coming soon!

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

Comprehensive documentation is available [here](https://rlinf.readthedocs.io/en/latest/).

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

| Type             | Status |
| :--------------: | :----: |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions!  See the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) for details.

## Citation and Acknowledgements

If you use RLinf, please cite the GitHub repository:

```bibtex
@misc{RLinf_repo,
  title        = {RLinf: Reinforcement Learning Infrastructure for Agentic AI},
  howpublished = {\url{https://github.com/RLinf/RLinf}},
  note         = {GitHub repository},
  year         = {2025}
}
```

**Paper:** A full paper describing RLinf will be released by **September 20, 2025**. We will update this section with the official citation and BibTeX when they become available.

**Acknowledgements:**
RLinf draws inspiration and benefits from the open-source community.  Special thanks to VeRL, AReaL, Megatron-LM, SGLang, and PyTorch FSDP. Please open an issue or pull request if your project or contribution is missing.

**Contact:**

We are seeking Postdocs, PhD/Master's students, and interns.  Contact us to join the team:

*   Chao Yu: zoeyuchao@gmail.com
*   Yu Wang: yu-wang@tsinghua.edu.cn