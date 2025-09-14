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

# RLinf: Revolutionizing Reinforcement Learning Infrastructure for Agentic AI

**RLinf is a cutting-edge, open-source infrastructure designed to empower the next generation of AI by providing flexible and scalable reinforcement learning capabilities.**  ([See the original repository](https://github.com/RLinf/RLinf))

## Key Features

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Execute macro-level logical flows through micro-level execution, decoupling workflow construction from physical communication and scheduling for enhanced flexibility and efficiency.
*   **Flexible Execution Modes:**
    *   **Collocated Mode:** Share all GPUs across all workers.
    *   **Disaggregated Mode:** Enables fine-grained pipelining for optimized resource utilization.
    *   **Hybrid Mode:** A customizable combination of collocated and disaggregated modes for tailored performance.
*   **Intelligent Auto-scheduling:** Automatically selects the optimal execution mode based on the training workload, eliminating manual resource allocation and streamlining the development process.
*   **Embodied Agent Support:**
    *   **Fast Adaptation:** Streamlined integration with leading VLA models such as OpenVLA, OpenVLA-OFT, and π₀.
    *   **Simulator Compatibility:** Supports mainstream CPU & GPU-based simulators through standardized RL interfaces (ManiSkill3, LIBERO).
    *   **First RL Fine-tuning:** Facilitates the first RL fine-tuning of the π₀ model family with a flow-matching action expert.
*   **Exceptional Performance:**
    *   **High Throughput:** Hybrid mode with fine-grained pipelining achieves **120%+** throughput improvement compared to alternative frameworks.
    *   **Dynamic Scaling:** The Automatic Online Scaling Strategy dynamically adjusts training resources, switching GPUs in seconds, and improving efficiency by 20-40% while preserving the on-policy nature of RL algorithms.
*   **Versatile Integration:**
    *   **Multiple Backend Integrations:**
        *   **FSDP + Hugging Face:** Ideal for rapid adaptation, new models, and beginners.
        *   **Megatron + SGLang:** Optimized for large-scale training, providing maximum efficiency for expert users.
    *   **Adaptive Communication:** Leverages an asynchronous communication channel for seamless data exchange.
    *   **Built-in RL Support:** Includes support for popular RL methods, including PPO, GRPO, DAPO, Reinforce++, and others.

## Roadmap

### System-Level Enhancements
-   [ ] Support for heterogeneous GPUs
-   [ ] Support for asynchronous pipeline execution
-   [ ] Support for Mixture of Experts (MoE)
-   [ ] Support for vLLM inference backend

### Application-Level Extensions
-   [ ] Support for Vision-Language Models (VLMs) training
-   [ ] Support for deep searcher agent training
-   [ ] Support for multi-agent training
-   [ ] Support for integration with more embodied simulators (e.g., Meta-World, GENESIS)
-   [ ] Support for more Vision Language Action models (VLAs), such as GR00T
-   [ ] Support for world model
-   [ ] Support for real-world RL embodied intelligence

## Getting Started

Explore the complete documentation [**Here**](https://rlinf.readthedocs.io/en/latest/).

**Quickstart Guide:**

*   [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
*   [Quickstart 1: PPO Training of VLAs on Maniskill3](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
*   [Quickstart 2: GRPO Training of LLMs on MATH](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
*   [Multi-node Training](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
*   [Model Evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)

**Key Design Elements:**

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

| Type             | Status                                                                                                                               |
| :--------------: | :------------------------------------------------------------------------------------------------------------------------------------- |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions to RLinf. Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before contributing.

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
-   Chao Yu: zoeyuchao@gmail.com
-   Yu Wang: yu-wang@tsinghua.edu.cn