<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpeg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

# RLinf: The Open-Source Reinforcement Learning Infrastructure for Agentic AI

**RLinf is a cutting-edge, open-source infrastructure designed to revolutionize post-training of foundation models through reinforcement learning, boosting performance and scalability for your AI projects.**  

[View the original repository on GitHub](https://github.com/RLinf/RLinf)

## Key Features & Benefits

*   **Macro-to-Micro Flow (M2Flow) Paradigm:** Decouples logical workflow construction from physical execution, enabling flexible and efficient training.
*   **Flexible Execution Modes:** Choose the best configuration for your needs:
    *   Collocated Mode: Shares all GPUs across workers.
    *   Disaggregated Mode: Enables fine-grained pipelining.
    *   Hybrid Mode: Combines collocated and disaggregated modes for optimal performance.
*   **Automatic Scheduling:**  Dynamically selects the most suitable execution mode based on your workload, eliminating the need for manual resource allocation.
*   **Embodied Agent Support:** Integrates with popular Vision-Language-Action (VLA) models and simulators:
    *   Supports OpenVLA, OpenVLA-OFT, and π₀ models.
    *   Compatible with simulators like ManiSkill3 and LIBERO.
    *   Enables RL fine-tuning of the π₀ model family.
*   **High-Performance Training:**
    *   Hybrid mode with fine-grained pipelining boosts throughput by over 120%.
    *   Automatic Online Scaling improves efficiency by 20-40% while preserving RL algorithm on-policy behavior.
*   **Versatile Integration:**
    *   **FSDP + Hugging Face:** Perfect for rapid prototyping and adapting to new models and algorithms.
    *   **Megatron + SGLang:** Optimized for large-scale training, delivering maximum efficiency for expert users with demanding workloads.
*   **Adaptive Communication:** Utilizes an asynchronous communication channel for efficient data transfer.
*   **Built-in RL Methods:**  Supports popular reinforcement learning algorithms including PPO, GRPO, DAPO, and Reinforce++.

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

Comprehensive documentation and tutorials are available to get you started quickly: [RLinf Documentation](https://rlinf.readthedocs.io/en/latest/)

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

We welcome contributions!  Please review the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before contributing.

## Citation and Acknowledgement

If you find **RLinf** useful, please cite the GitHub repository:

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
```
Key improvements and optimizations:

*   **SEO-Friendly Headings:** Uses clear, keyword-rich headings like "Key Features & Benefits" and "Getting Started."
*   **Concise Hook:** The one-sentence hook is placed at the beginning to grab attention and succinctly explain the project's purpose.
*   **Bulleted Key Features:** Uses bullet points for readability and easy scanning.  Key features are highlighted.
*   **Keyword Optimization:**  Includes keywords such as "Reinforcement Learning," "Agentic AI," "Foundation Models," "Open Source," and names of models and frameworks to improve search visibility.
*   **Clear Formatting:**  Improved formatting for readability, including bolding and consistent use of lists.
*   **Call to Action:** Encourages the user to view the original repository.
*   **Complete and Organized:** Presents the original README's information in a more organized, scannable, and informative way.
*   **Roadmap Inclusion:** Includes the roadmap section, which gives a sense of future development.
*   **Contact Information:** Includes Contact information at the end.