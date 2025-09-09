<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
  <a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
  <a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
  <a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpeg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<h1 align="center">
  RLinf: Unleash the Power of Agentic AI with Scalable Reinforcement Learning Infrastructure
</h1>

RLinf (**R**einforcement **L**earning **Inf**rastructure) is a cutting-edge, open-source framework designed to supercharge the post-training of foundation models using reinforcement learning, paving the way for the next generation of intelligent agents.  For more details, visit the original repository: [RLinf on GitHub](https://github.com/RLinf/RLinf).

## Key Features of RLinf

RLinf empowers researchers and developers with a robust and flexible infrastructure to build and train advanced agentic AI models.

*   **Macro-to-Micro Flow (M2Flow) Paradigm:**
    *   Decouples logical workflow construction (programmable) from physical communication and scheduling (efficiency), enabling streamlined development.
*   **Flexible Execution Modes:**
    *   **Collocated Mode:** Shares all GPUs across all workers.
    *   **Disaggregated Mode:** Enables fine-grained pipelining.
    *   **Hybrid Mode:** Combines collocated and disaggregated modes, offering customizable resource utilization.
*   **Auto-Scheduling Strategy:**
    *   Dynamically selects the most efficient execution mode based on the training workload, eliminating the need for manual resource allocation.
*   **Embodied Agent Support:**
    *   Supports fast adaptation for various VLA models: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [π₀](https://github.com/Physical-Intelligence/openpi).
    *   Integrates with mainstream CPU & GPU-based simulators: [ManiSkill3](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
    *   Enables pioneering RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.
*   **Optimized for Speed and Efficiency:**
    *   **Hybrid Mode + Fine-Grained Pipelining:** Achieves over **120%** throughput improvement compared to other frameworks.
    *   **Automatic Online Scaling Strategy:** Dynamically scales resources, with GPU switching in seconds, improving efficiency by 20–40% while maintaining on-policy RL algorithm behavior.
*   **Ease of Use and Flexibility:**
    *   **Multiple Backend Integrations:**
        *   **FSDP + Hugging Face:** Ideal for rapid model and algorithm adaptation, making it perfect for beginners and fast prototyping.
        *   **Megatron + SGLang:** Optimized for large-scale training, delivering maximum efficiency for expert users.
    *   Adaptive communication through an asynchronous communication channel.
    *   Built-in support for popular RL algorithms like [PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476), [Reinforce++](https://arxiv.org/abs/2501.03262), and more.

## Roadmap

### 1. System-Level Enhancements

*   \[ ] Support for heterogeneous GPUs
*   \[ ] Support for asynchronous pipeline execution
*   \[ ] Support for Mixture of Experts (MoE)
*   \[ ] Support for vLLM inference backend

### 2. Application-Level Extensions

*   \[ ] Support for Vision-Language Models (VLMs) training
*   \[ ] Support for deep searcher agent training
*   \[ ] Support for multi-agent training
*   \[ ] Support for integration with more embodied simulators (e.g., [Meta-World](https://github.com/Farama-Foundation/Metaworld), [GENESIS](https://github.com/Genesis-Embodied-AI/Genesis))
*   \[ ] Support for more Vision Language Action models (VLAs), such as [GR00T](https://github.com/NVIDIA/Isaac-GR00T)
*   \[ ] Support for world model
*   \[ ] Support for real-world RL embodied intelligence

## Getting Started

Comprehensive documentation is available [**Here**](https://rlinf.readthedocs.io/en/latest/).

**Quickstart Guides:**

*   [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
*   [Quickstart 1: PPO Training of VLAs on Maniskill3](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
*   [Quickstart 2: GRPO Training of LLMs on MATH](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
*   [Multi-node Training](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
*   [Model Evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)

**Key Design Concepts:**

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

**Extending the Framework:**

*   [Adding new Environments](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html)
*   [Adding new Models with FSDP+Huggingface backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_fsdp.html)
*   [Adding new Models with Megatron+SGLang backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_megatron.html)

**Blogs:**

*   [Comparison with VeRL](https://rlinf.readthedocs.io/en/latest/rst_source/blog/compare_with_verl.html)

## Build Status

| Type              | Status                                                                                                       |
| :---------------: | :-------------------------------------------------------------------------------------------------------------: |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

Contributions to RLinf are welcome!  Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before submitting any work.

## Citation and Acknowledgements

If you find **RLinf** valuable, please cite the GitHub repository:

```bibtex
@misc{RLinf_repo,
  title        = {RLinf: Reinforcement Learning Infrastructure for Agentic AI},
  howpublished = {\url{https://github.com/RLinf/RLinf}},
  note         = {GitHub repository},
  year         = {2025}
}
```

**Paper:** A full paper describing RLinf will be released by **September 20, 2025**. This section will be updated with the complete citation and BibTeX information upon release.

**Acknowledgements**

RLinf is built upon the foundation of the open-source community. We are grateful for the contributions and inspiration from the teams and developers of VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP).  If your project or contribution has been inadvertently omitted, please submit an issue or pull request so we can properly acknowledge it.

**Contact:**

We encourage applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!

*   Chao Yu: zoeyuchao@gmail.com
*   Yu Wang: yu-wang@tsinghua.edu.cn
```
Key improvements and rationale:

*   **SEO Optimization:**  Includes keywords like "Reinforcement Learning", "Agentic AI", "Infrastructure", "Foundation Models", "Scalable".  Uses headings and clear descriptions to improve searchability.
*   **Clear Hook:** Starts with a compelling one-sentence hook to grab the reader's attention.
*   **Structured Content:** Uses headings, subheadings, and bullet points for readability and easy scanning.
*   **Concise Language:**  Streamlines descriptions to be more direct and to the point.
*   **Key Feature Focus:** Highlights the most important features, with brief descriptions, including their benefits.
*   **Complete Sections:** Keeps all original sections while improving their organization and clarity.
*   **Actionable Links:**  Makes links clear.
*   **Updated for 2024 (but with 2025 paper release note)**: The original `README` implied it was just released.
*   **Removed the TODO:** The original TODO-link section was removed, and this README is ready to publish.
*   **Improved Contribution & Contact:**  Clearer contribution guidelines and contact info.