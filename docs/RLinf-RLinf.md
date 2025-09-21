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

# RLinf: Powering Agentic AI with Flexible Reinforcement Learning Infrastructure

**RLinf is an open-source infrastructure designed to revolutionize post-training foundation models through scalable and efficient reinforcement learning.**  ([See the original repo](https://github.com/RLinf/RLinf))

## Key Features

RLinf is a cutting-edge framework with the following advantages:

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Decouples logical workflow design (programmable) from physical execution and scheduling (efficient).
*   **Flexible Execution Modes:**
    *   **Collocated Mode:** Efficient use of all GPUs across workers.
    *   **Disaggregated Mode:** Enables fine-grained pipelining for optimized performance.
    *   **Hybrid Mode:** Combines collocated and disaggregated modes for maximum flexibility.
*   **Automated Scheduling:** Dynamically selects the optimal execution mode, eliminating manual resource allocation.
*   **Embodied Agent Support:**
    *   Fast adaptation for leading VLA models like OpenVLA, OpenVLA-OFT, and π₀.
    *   Standardized RL interfaces for CPU & GPU-based simulators (ManiSkill3, LIBERO).
    *   First RL fine-tuning support for the π₀ model with a flow-matching action expert.
*   **High Performance:**
    *   **120%+** throughput improvement with Hybrid mode and fine-grained pipelining.
    *   **Automatic Online Scaling:** Dynamically scales training resources with GPU switching in seconds, boosting efficiency by 20–40%.
*   **Ease of Use & Integration:**
    *   Multiple Backend Integrations including FSDP + Hugging Face and Megatron + SGLang.
    *   Asynchronous Communication Channel for adaptive communication.
    *   Built-in support for popular RL methods (PPO, GRPO, DAPO, Reinforce++, and more).

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
*   [ ] Support for integration with more embodied simulators (e.g., Meta-World, GENESIS)
*   [ ] Support for more Vision Language Action models (VLAs), such as GR00T
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

| Type             | Status                                                                                                                                |
| :--------------: | :------------------------------------------------------------------------------------------------------------------------------------ |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions! Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before contributing.

## Citation and Acknowledgement

If you use **RLinf**, please cite the GitHub repository:

```bibtex
@misc{RLinf_repo,
  title        = {RLinf: Reinforcement Learning Infrastructure for Agentic AI},
  howpublished = {\url{https://github.com/RLinf/RLinf}},
  note         = {GitHub repository},
  year         = {2025}
}
```

**Paper:** A full paper describing RLinf will be released by **September 20, 2025**.  We will update this section with the official citation and BibTeX when they become available.

**Acknowledgements**
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn
```
Key improvements and SEO optimizations:

*   **Concise Title:** "RLinf: Powering Agentic AI with Flexible Reinforcement Learning Infrastructure" is more descriptive and keyword-rich.
*   **One-Sentence Hook:** The introductory sentence immediately grabs attention and explains the project's purpose.
*   **Keyword Optimization:** The README uses relevant keywords throughout (Reinforcement Learning, Agentic AI, Infrastructure, Foundation Models, etc.).
*   **Clear Headings:**  Uses clear and consistent headings (e.g., "Key Features," "Roadmap," "Getting Started").
*   **Bulleted Lists:**  Features and other key information are presented using bulleted lists, making them easy to scan.
*   **Benefit-Driven Descriptions:** The descriptions of features focus on the benefits to the user (e.g., "High Performance" and "Ease of Use").
*   **Call to Action:** Includes calls to action (e.g., "Getting Started," "Contribution Guidelines").
*   **SEO-Friendly Formatting:** Uses Markdown formatting for better readability and search engine indexing.
*   **Internal Links:**  Links to relevant documentation sections within the README.
*   **Concise Language:** Avoids overly verbose language.
*   **Includes Contact Information:** Keeps the "Contact" section.
*   **Clear Citation Instructions:** Provides the correct citation format.