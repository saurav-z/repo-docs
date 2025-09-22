<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<!-- <a href="TODO"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a> -->
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpeg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

<h1 align="center">
  RLinf: Supercharge Your Agentic AI with Scalable Reinforcement Learning
</h1>

RLinf is a cutting-edge, open-source infrastructure designed to revolutionize the training of post-training foundation models using reinforcement learning. This is the perfect framework for researchers and developers looking to push the boundaries of agentic AI.  <a href="https://github.com/RLinf/RLinf">Explore the RLinf repository here!</a>

## Key Features of RLinf

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Experience a new approach that decouples logical workflow construction (programmable) from physical communication and scheduling (efficiency), allowing for complex workflows while optimizing performance.

*   **Flexible Execution Modes:** Choose the best approach for your needs with:
    *   **Collocated Mode:** Shares all GPUs across all workers.
    *   **Disaggregated Mode:** Enables fine-grained pipelining for advanced optimization.
    *   **Hybrid Mode:** Combine collocated and disaggregated modes for a custom configuration.

*   **Automatic Scheduling:**  Let RLinf intelligently select the most suitable execution mode, eliminating the need for manual resource allocation and simplifying your workflow.

*   **Embodied Agent Support:**
    *   Seamlessly integrates with popular VLA models like [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [π₀](https://github.com/Physical-Intelligence/openpi).
    *   Supports mainstream CPU & GPU-based simulators via standardized RL interfaces: [ManiSkill3](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
    *   Enables the first RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.

*   **High-Performance Training:**
    *   Hybrid mode with fine-grained pipelining results in **120%+** throughput improvements compared to other frameworks.
    *   Automatic Online Scaling Strategy dynamically adjusts training resources, providing efficiency gains of 20–40% while preserving the on-policy nature of RL algorithms.

*   **Flexible and Easy-to-Use Integrations:**
    *   **Multiple Backend Support:** Offers both FSDP + Hugging Face and Megatron + SGLang integrations for maximum flexibility.
    *   **Adaptive Communication:** Leverages an asynchronous communication channel for efficient data transfer.
    *   **Built-in RL Methods:** Supports leading RL methods like PPO, GRPO, DAPO, Reinforce++, and more, making it easy to implement your research or project.

## Roadmap

### 1. System-Level Enhancements
*   [ ] Support for heterogeneous GPUs
*   [ ] Support for asynchronous pipeline execution
*   [ ] Support for Mixture of Experts (MoE)
*   [ ] Support for vLLM inference backend

### 2. Application-Level Extensions
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

**Key Design Resources:**

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

**Blogs**

*   [Comparison with VeRL](https://rlinf.readthedocs.io/en/latest/rst_source/blog/compare_with_verl.html)

## Build Status

| Type              | Status                                                                                                                                   |
| :---------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions!  Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before getting started.

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
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn