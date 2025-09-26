<div align="center">
  <img src="docs/source-en/_static/svg/logo_white.svg" alt="RLinf-logo" width="600"/>
</div>

<div align="center">
<a href="https://arxiv.org/abs/2509.15965"><img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv"></a>
<a href="https://huggingface.co/RLinf"><img src="https://img.shields.io/badge/HuggingFace-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
<a href="https://rlinf.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-Purple?color=8A2BE2&logo=readthedocs"></a>
<a href="https://rlinf.readthedocs.io/zh-cn/latest/"><img src="https://img.shields.io/badge/中文文档-red?logo=readthedocs"></a>
<a href="https://deepwiki.com/RLinf/RLinf"><img src="https://img.shields.io/badge/Ask%20DeepWiki-1DA1F2?logo=databricks&logoColor=white&color=00ADEF" alt="Ask DeepWiki"></a>
<a href="https://github.com/RLinf/misc/blob/main/pic/wechat.jpg?raw=true"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp"></a>
</div>

## RLinf: Your Foundation for Agentic AI

**RLinf (Reinforcement Learning Infrastructure) is an open-source powerhouse designed to supercharge your post-training foundation models with reinforcement learning, enabling advanced AI capabilities.**  Explore the [original repository](https://github.com/RLinf/RLinf) for more details.

**Key Features:**

*   **Macro-to-Micro Flow (M2Flow):** Decouples logical workflow construction (programmable) from physical communication and scheduling (efficient) for optimal performance.
*   **Flexible Execution Modes:** Offers collocated, disaggregated, and hybrid modes for customized resource utilization.
*   **Auto-Scheduling Strategy:** Automatically selects the most efficient execution mode, simplifying training and resource management.
*   **Embodied Agent Support:**  Ready-to-use with popular VLA models like OpenVLA, and simulators (ManiSkill3, LIBERO).
*   **Fast Training:** Hybrid mode and automatic online scaling for significant throughput improvements (120%+) and rapid GPU switching.
*   **Multiple Backend Integrations:** Supports FSDP + Hugging Face for easy model adaptation and Megatron + SGLang for large-scale, efficient training.
*   **Built-in RL Methods:**  Includes PPO, GRPO, DAPO, Reinforce++, and more for diverse applications.

## Main Results

### Embodied Intelligence

**(See table in the original README for detailed performance data)**

RLinf delivers state-of-the-art performance, especially when fine-tuning Vision-Language-Action models with GRPO, achieving impressive gains.

### Math Reasoning

**(See table in the original README for detailed performance data)**

RLinf pushes the boundaries of math reasoning, consistently surpassing existing models on benchmarks such as AIME and GPQA-diamond.

## Roadmap

*   **System-Level Enhancements:** Support for heterogeneous GPUs, asynchronous pipeline execution, Mixture of Experts (MoE), and vLLM inference backend.
*   **Application-Level Extensions:** Vision-Language Models (VLMs) training, deep searcher agent training, multi-agent training, and integration with more embodied simulators and VLA models, world models, and real-world RL.

## Getting Started

*   Comprehensive documentation is available [**Here**](https://rlinf.readthedocs.io/en/latest/).

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

| Type             | Status |
| :--------------: | :----: |
| Reasoning RL-MATH | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/math_e2e.yml) |
| Embodied RL-VLA   | [![Build Status](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml/badge.svg)](https://github.com/RLinf/RLinf/actions/workflows/embodied_e2e.yml) |

## Contribution Guidelines

We welcome contributions to RLinf. Please read the [contribution guide](https://rlinf.readthedocs.io/en/latest/index.html#contribution-guidelines) before contributing.

## Citation and Acknowledgement

If you find **RLinf** helpful, please cite the paper:

```bibtex
@misc{yu2025rlinfflexibleefficientlargescale,
  title={RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation}, 
  author={Chao Yu and Yuanqing Wang and Zhen Guo and Hao Lin and Si Xu and Hongzhi Zang and Quanlu Zhang and Yongji Wu and Chunyang Zhu and Junhao Hu and Zixiao Huang and Mingjie Wei and Yuqing Xie and Ke Yang and Bo Dai and Zhexuan Xu and Xiangyuan Wang and Xu Fu and Zhihao Liu and Kang Chen and Weilin Liu and Gang Liu and Boxun Li and Jianlei Yang and Zhi Yang and Guohao Dai and Yu Wang},
  year={2025},
  eprint={2509.15965},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.15965}, 
}
```

If you use RL+VLA in RLinf, you can also cite our empirical study paper:
```bibtex
@misc{liu2025rlbringvlageneralization,
  title={What Can RL Bring to VLA Generalization? An Empirical Study}, 
  author={Jijia Liu and Feng Gao and Bingwen Wei and Xinlei Chen and Qingmin Liao and Yi Wu and Chao Yu and Yu Wang},
  year={2025},
  eprint={2505.19789},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.19789}, 
}
```

**Acknowledgements**
RLinf has been inspired by, and benefits from, the ideas and tooling of the broader open-source community.
In particular, we would like to thank the teams and contributors behind VeRL, AReaL, Megatron-LM, SGLang, and PyTorch Fully Sharded Data Parallel (FSDP), and if we have inadvertently missed your project or contribution, please open an issue or a pull request so we can properly credit you.

**Contact:**
We welcome applications from Postdocs, PhD/Master's students, and interns. Join us in shaping the future of RL infrastructure and embodied AI!
- Chao Yu: zoeyuchao@gmail.com
- Yu Wang: yu-wang@tsinghua.edu.cn