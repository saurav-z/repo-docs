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

# RLinf: The Cutting-Edge Infrastructure for Agentic AI & Reinforcement Learning

**RLinf is an open-source infrastructure designed to revolutionize post-training foundation models through reinforcement learning, offering unprecedented flexibility, scalability, and efficiency.** ([See the original repo](https://github.com/RLinf/RLinf))

## Key Features

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Decouples logical workflow construction from physical communication and scheduling, enabling programmable workflows and efficient execution.
*   **Flexible Execution Modes:** Offers collocated, disaggregated, and hybrid modes for optimal resource utilization and fine-grained pipelining.
*   **Intelligent Auto-Scheduling:** Automatically selects the most suitable execution mode, eliminating the need for manual resource allocation.
*   **Embodied Agent Support:**  Provides fast adaptation for popular VLA models (OpenVLA, OpenVLA-OFT, π₀) and supports mainstream simulators (ManiSkill3, LIBERO) with standardized RL interfaces.
*   **High Performance:** Achieves over **120%+** throughput improvement with hybrid mode and fine-grained pipelining, with dynamic scaling that improves efficiency by 20-40% while preserving the on-policy nature of RL algorithms.
*   **Multiple Backend Integrations:** Seamlessly integrates with FSDP + Hugging Face for rapid prototyping and Megatron + SGLang for large-scale training.
*   **Adaptive Communication:** Leverages an asynchronous communication channel for efficient data transfer.
*   **Built-in RL Methods:** Supports popular RL algorithms such as PPO, GRPO, DAPO, and Reinforce++.

## What's New

*   [2025/08] RLinf is open-sourced. The formal v0.1 will be released soon. The paper [RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation]() will also be released accordingly.

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

**Quickstart:**

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

| Type            | Status                                                                                                                                                                     |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml)                   |
| Embodied RL-VLA  | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml)            |

## Contribution Guidelines

We welcome contributions!  See the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) for details.

## Citation and Acknowledgement

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
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**

We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!

*   Chao Yu: zoeyuchao@gmail.com
*   Yu Wang: yu-wang@tsinghua.edu.cn
```
Key improvements and explanations:

*   **SEO-Optimized Title:**  Replaced the generic title with a more descriptive and keyword-rich title, optimized for search engines.
*   **One-Sentence Hook:** Added a concise and compelling opening sentence to immediately grab the reader's attention and convey the core value proposition.
*   **Clear Headings:**  Organized the information with clear, descriptive headings and subheadings for readability and SEO benefits (helps search engines understand the content).
*   **Bulleted Key Features:**  Used bullet points to highlight key features, making the information easy to scan and digest.  Keywords related to features are included.
*   **Emphasis on Benefits:**  Focused on what RLinf *does* for users (e.g., "revolutionize," "unprecedented flexibility," "optimal resource utilization") instead of just listing features.
*   **Concise Language:**  Streamlined the wording to improve clarity and conciseness.
*   **Internal Linking:**  Included links to relevant sections within the documentation, improving user experience and SEO.
*   **Call to Action (Implied):** The entire structure of the README encourages users to explore the project.
*   **Keywords:** Used relevant keywords like "Reinforcement Learning," "Agentic AI," "Infrastructure," "Foundation Models," "Training," "Scalability," "Efficiency", "Embodied AI" throughout the document.
*   **Clean formatting:** Maintaining the original formatting, ensuring readability.