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
  RLinf: The Open-Source Foundation for Advanced Reinforcement Learning and Agentic AI
</h1>

RLinf is a cutting-edge, open-source infrastructure designed to empower the development of agentic AI through efficient and scalable reinforcement learning. Built for post-training foundation models, RLinf provides a robust backbone for the next generation of AI, offering flexibility, speed, and ease of use.  [Explore the RLinf project on GitHub](https://github.com/RLinf/RLinf).

## Key Features of RLinf

RLinf distinguishes itself with innovative features, designed for both researchers and developers:

*   **Macro-to-Micro Flow (M2Flow):** A novel paradigm that decouples logical workflow construction from physical communication and scheduling for optimal efficiency. This unique approach allows for flexible and programmable macro-level flows with micro-level execution.
*   **Flexible Execution Modes:**
    *   **Collocated Mode:** Shares all GPUs across all workers.
    *   **Disaggregated Mode:** Enables fine-grained pipelining.
    *   **Hybrid Mode:** Combines collocated and disaggregated modes for customized resource utilization.
*   **Auto-scheduling Strategy:** Dynamically selects the most suitable execution mode based on training workload, eliminating manual resource allocation and maximizing performance.
*   **Embodied Agent Support:**
    *   Fast adaptation support for mainstream VLA models: [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft), and [π₀](https://github.com/Physical-Intelligence/openpi).
    *   Support for mainstream CPU & GPU-based simulators via standardized RL interfaces: [ManiSkill3](https://github.com/haosulab/ManiSkill), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).
    *   Enabling the first RL fine-tuning of the $\pi_0$ model family with a flow-matching action expert.
*   **Superior Speed & Efficiency:**
    *   Hybrid mode with fine-grained pipelining achieves a **120%+** throughput improvement compared to other frameworks.
    *   Automatic Online Scaling Strategy: dynamically scales training resources, with GPU switching completed within seconds, further improving efficiency by 20–40% while preserving the on-policy nature of RL algorithms.
*   **User-Friendly Integration:**
    *   Multiple Backend Integrations:
        *   FSDP + Hugging Face: Rapid adaptation to new models and algorithms, ideal for beginners and fast prototyping.
        *   Megatron + SGLang: Optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.
    *   Adaptive communication via the asynchronous communication channel.
    *   Built-in support for popular RL methods, including [PPO](https://arxiv.org/abs/1707.06347), [GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476), [Reinforce++](https://arxiv.org/abs/2501.03262), and more.

## Getting Started

Dive into RLinf with comprehensive documentation and resources:

*   **Documentation:** [Official Documentation](https://rlinf.readthedocs.io/en/latest/)
*   **Quickstarts:**
    *   [Installation](https://rlinf.readthedocs.io/en/latest/rst_source/start/installation.html)
    *   [Quickstart 1: PPO Training of VLAs on Maniskill3](https://rlinf.readthedocs.io/en/latest/rst_source/start/vla.html)
    *   [Quickstart 2: GRPO Training of LLMs on MATH](https://rlinf.readthedocs.io/en/latest/rst_source/start/llm.html)
    *   [Multi-node Training](https://rlinf.readthedocs.io/en/latest/rst_source/start/distribute.html)
    *   [Model Evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/start/eval.html)
*   **Key Design Principles:**
    *   [Unified User Interface Usage](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/user/index.html)
    *   [Flexible Execution Modes](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/mode/index.html)
    *   [Enable Automatic Scheduling](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/scheduler/index.html)
    *   [Elastic Communication](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/communication/index.html)
*   **Example Gallery:**
    *   [Embodied Intelligence Vision-Language-Action Model training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied.html)
    *   [Math Reasoning Model Training](https://rlinf.readthedocs.io/en/latest/rst_source/examples/reasoning.html)
*   **Advanced Features:**
    *   [5D Parallelism Configuration for Megatron-LM](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/5D.html)
    *   [LoRA Integration for efficient fine-tuning](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/lora.html)
    *   [Switch between different versions of SGLang](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/version.html)
    *   [Checkpoint Resume and Recovery Support](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/advance/resume.html)
*   **Extending The Framework:**
    *   [Adding new Environments](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_env.html)
    *   [Adding new Models with FSDP+Huggingface backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_fsdp.html)
    *   [Adding new Models with Megatron+SGLang backend](https://rlinf.readthedocs.io/en/latest/rst_source/tutorials/extend/new_model_megatron.html)
*   **Blogs:**
    *   [Comparison with VeRL](https://rlinf.readthedocs.io/en/latest/rst_source/blog/compare_with_verl.html)

## Roadmap

RLinf is continuously evolving, with these key areas of development planned:

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

## Build Status

| Type             | Status |
| :--------------: | :----: |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

RLinf welcomes contributions from the community.  Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before submitting any contributions.

## Citation and Acknowledgement

If you utilize **RLinf** in your research, please cite the GitHub repository:

```bibtex
@misc{RLinf_repo,
  title        = {RLinf: Reinforcement Learning Infrastructure for Agentic AI},
  howpublished = {\url{https://github.com/RLinf/RLinf}},
  note         = {GitHub repository},
  year         = {2025}
}
```

**Paper:** A full paper describing RLinf will be released by **September 20, 2025**. We will update this section with the official citation and BibTeX when they become available.

**Acknowledgements**
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn
```
Key improvements and SEO considerations:

*   **Compelling Hook:**  "RLinf: The Open-Source Foundation for Advanced Reinforcement Learning and Agentic AI" immediately grabs attention and sets the project's purpose.
*   **Strategic Keywords:** Used terms like "Reinforcement Learning," "Agentic AI," "open-source," "infrastructure," and "foundation models" throughout the README, which are crucial for SEO.
*   **Clear Headings:**  Organized content with descriptive headings for better readability and searchability.
*   **Concise Bullet Points:**  Emphasized key features using bullet points for easy comprehension.
*   **Actionable Links:** Kept important links visible, making it easier for users to navigate and explore.
*   **Focus on Benefits:** Highlighted what RLinf *does* (enables, empowers, provides), focusing on user value.
*   **Call to Action:** Encourages exploration and contribution with clear language.
*   **Updated Roadmap:** Added the roadmap and key features to ensure that the features are visible.