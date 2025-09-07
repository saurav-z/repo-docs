<div align="center">
  <img src="docs/source/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
  <a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
  <a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
  <a href="https://github.com/Lin-xs/RLinf-community/blob/main/wechat-QR.jpeg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<h1 align="center">
  RLinf: Unleash the Power of Reinforcement Learning for Agentic AI
</h1>

RLinf is a cutting-edge, open-source reinforcement learning infrastructure designed to revolutionize the training of agentic AI models. This flexible and scalable platform empowers researchers and developers to push the boundaries of AI, supporting everything from large language models to embodied agents.

**[Explore the RLinf GitHub Repository](https://github.com/RLinf/RLinf)**

## Key Features of RLinf

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Decouples logical workflow construction from physical execution, enabling flexible and efficient training.
*   **Flexible Execution Modes:** Supports Collocated, Disaggregated, and Hybrid modes for optimal resource utilization and performance.
*   **Automatic Scheduling:** Intelligently selects the best execution mode based on the workload, eliminating manual configuration.
*   **Embodied Agent Support:**
    *   Seamless integration with popular VLA models (OpenVLA, OpenVLA-OFT, π₀).
    *   Compatibility with CPU & GPU-based simulators (ManiSkill3, LIBERO).
    *   Enabling the first RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.
*   **High Performance:**
    *   Hybrid mode with fine-grained pipelining boosts throughput by over **120%** compared to other frameworks.
    *   Dynamic Online Scaling Strategy increases efficiency by **20–40%**.
*   **Easy to Use and Adapt:**
    *   Multiple Backend Integrations: FSDP + Hugging Face for rapid prototyping and Megatron + SGLang for large-scale training.
    *   Adaptive communication via the asynchronous communication channel.
    *   Built-in support for popular RL methods (PPO, GRPO, DAPO, Reinforce++, etc.).

## Roadmap

*   **System-Level Enhancements**
    *   Support for heterogeneous GPUs
    *   Support for asynchronous pipeline execution
    *   Support for Mixture of Experts (MoE)
    *   Support for vLLM inference backend

*   **Application-Level Extensions**
    *   Support for Vision-Language Models (VLMs) training
    *   Support for deep searcher agent training
    *   Support for multi-agent training
    *   Support for integration with more embodied simulators (e.g., Meta-World, GENESIS)
    *   Support for more Vision Language Action models (VLAs), such as GR00T
    *   Support for world model
    *   Support for real-world RL embodied intelligence

## Getting Started

Dive into RLinf with our comprehensive documentation and quickstart guides:

*   [**Documentation**](https://rlinf.readthedocs.io/en/latest/)
    *   [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
    *   [Quickstart 1: PPO Training of VLAs on Maniskill3](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
    *   [Quickstart 2: GRPO Training of LLMs on MATH](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
    *   [Multi-node Training](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
    *   [Model Evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)

**Key Design Tutorials:**
  * [Unified User Interface Usage](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/index.html)
  * [Flexible Execution Modes](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html)
  * [Enable Automatic Scheduling](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/scheduler/index.html)
  * [Elastic Communication](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/communication/index.html)

**Example Gallery:**

*   [Embodied Intelligence Vision-Language-Action Model training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied.html)
*   [Math Reasoning Model Training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html)

**Advanced Features Tutorials:**

*   [5D Parallelism Configuration for Megatron-LM](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html)
*   [LoRA Integration for efficient fine-tuning](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/lora.html)
*   [Switch between different versions of SGLang](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/version.html)
*   [Checkpoint Resume and Recovery Support](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html)

**Extending The Framework Tutorials:**

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

We welcome contributions!  Please read the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before getting started.

## Citation and Acknowledgements

If you find RLinf useful, please cite the GitHub repository:

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

RLinf benefits from and is inspired by the open-source community, particularly VeRL, AReaL, Megatron-LM, SGLang, and PyTorch FSDP.  If we've missed your project, please open an issue or pull request.

**Contact**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn