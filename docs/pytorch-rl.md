[![Unit-tests](https://github.com/pytorch/rl/actions/workflows/test-linux.yml/badge.svg)](https://github.com/pytorch/rl/actions/workflows/test-linux.yml)
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://pytorch.org/rl/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-blue.svg)](https://pytorch.github.io/rl/dev/bench/)
[![codecov](https://codecov.io/gh/pytorch/rl/branch/main/graph/badge.svg?token=HcpK1ILV6r)](https://codecov.io/gh/pytorch/rl)
[![Twitter Follow](https://img.shields.io/twitter/follow/torchrl1?style=social)](https://twitter.com/torchrl1)
[![Python version](https://img.shields.io/pypi/pyversions/torchrl.svg)](https://www.python.org/downloads/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pytorch/rl/blob/main/LICENSE)
<a href="https://pypi.org/project/torchrl"><img src="https://img.shields.io/pypi/v/torchrl" alt="pypi version"></a>
<a href="https://pypi.org/project/torchrl-nightly"><img src="https://img.shields.io/pypi/v/torchrl-nightly?label=nightly" alt="pypi nightly version"></a>
[![Downloads](https://static.pepy.tech/personalized-badge/torchrl?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads)](https://pepy.tech/project/torchrl)
[![Downloads](https://static.pepy.tech/personalized-badge/torchrl-nightly?period=total&units=international_system&left_color=blue&right_color=orange&left_text=Downloads%20(nightly))](https://pepy.tech/project/torchrl-nightly)
[![Discord Shield](https://dcbadge.vercel.app/api/server/cZs26Qq3Dd)](https://discord.gg/cZs26Qq3Dd)

# TorchRL: Your Gateway to Cutting-Edge Reinforcement Learning in PyTorch

TorchRL is an open-source library providing tools to build and train RL algorithms in PyTorch.  Explore the [original repo](https://github.com/pytorch/rl).

## Key Features:

*   🐍 **Python-First Design:** Simplifies RL development with Python's ease of use and flexibility.
*   ⏱️ **High-Performance:** Optimized for speed, ideal for research and production RL applications.
*   🧮 **Modular & Customizable:**  Easily adapt or create new components for your unique needs.
*   📚 **Comprehensive Documentation:** Learn quickly with thorough documentation and examples.
*   ✅ **Reliable & Stable:**  Built with rigorous testing for dependable performance.
*   ⚙️ **Reusable Functionals:** Access a suite of reusable functions for cost, returns, and data processing.
*   🤖 **LLM API:** A complete framework for language model fine-tuning including RLHF, supervised fine-tuning, and tool-augmented training.

## What's New: LLM API

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

- 🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
- 💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
- 🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
- 🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
- ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
- 🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

The LLM API follows TorchRL's modular design principles, allowing you to mix and match components for your specific use case. Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

## Core Design Principles

*   🔥 **PyTorch Ecosystem Alignment:** Follows the structure and conventions of PyTorch libraries.
*   ➖ **Minimal Dependencies:** Built with standard Python libraries, NumPy, and PyTorch; Optional dependencies for common environment libraries and datasets.

Read the [full paper](https://arxiv.org/abs/2306.00577) for a more curated description of the library.

##  Getting Started

Jumpstart your RL journey with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started)!

## Essential Components:

*   **[Environments](https://github.com/pytorch/rl/blob/main/torchrl/envs):** A unified interface for environments, including common libraries.
*   **[Data Collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py):** Efficient data collection with multiprocess and distributed support.
*   **[Replay Buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py):**  Modular replay buffers with storage options for offline RL.
*   **[Environment Transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py):**  Vectorized transforms for data preparation.
*   **[Models & Architectures](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/):**  Actor-critic and other common models.
*   **[Loss Modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives):**  Efficient loss calculations.
*   **[Trainers](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py):**  A generic trainer class for streamlined training loops.
*   **[Exploration Wrappers & Modules](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py):**  Easily switch between exploration and exploitation.
*   **[TensorDict](https://github.com/pytorch/tensordict/):** A convenient data structure that can be used to streamline one's
RL codebase

## Examples

Check out our [SOTA implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) for real-world algorithm examples.

## Documentation and Knowledge Base

*   Access full documentation [here](https://pytorch.org/rl).
*   Explore the RL knowledge base [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

## Citation

If you use TorchRL in your work, please cite:
```
@misc{bou2023torchrl,
      title={TorchRL: A data-driven decision-making library for PyTorch}, 
      author={Albert Bou and Matteo Bettini and Sebastian Dittert and Vikash Kumar and Shagun Sodhani and Xiaomeng Yang and Gianni De Fabritiis and Vincent Moens},
      year={2023},
      eprint={2306.00577},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Installation

Follow these steps to get TorchRL up and running:

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv torchrl
    source torchrl/bin/activate  # or venv\Scripts\activate on Windows
    ```
    or
    ```bash
    conda create --name torchrl python=3.9
    conda activate torchrl
    ```

2.  **Install Dependencies:**

    *   Ensure PyTorch is installed based on your needs (stable or nightly). Refer to [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

    *   Install TorchRL:
        ```bash
        pip3 install torchrl
        ```
        or
         ```bash
         # Install and build locally v0.8.1 of the library without cloning
         pip3 install git+https://github.com/pytorch/rl@v0.8.1
         # Clone the library and build it locally
         git clone https://github.com/pytorch/tensordict
         git clone https://github.com/pytorch/rl
         pip install -e tensordict
         pip install -e rl
         ```
        or for nightly:
        ```bash
        pip3 install tensordict-nightly torchrl-nightly
        ```

    *   Install optional dependencies:
        ```bash
        pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher
        pip3 install "moviepy<2.0.0"
        pip3 install dm_control
        pip3 install "gym[atari]" "gym[accept-rom-license]" pygame
        pip3 install pytest pyyaml pytest-instafail
        pip3 install tensorboard
        pip3 install wandb
        ```
## Contributing

Contributions are welcome! Review the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) and [open contributions](https://github.com/pytorch/rl/issues/509).