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

# TorchRL: A PyTorch Library for Reinforcement Learning

**TorchRL** empowers researchers and developers with a flexible, efficient, and modular toolkit for building and deploying reinforcement learning (RL) solutions. Explore the power of RL by visiting the [original repo](https://github.com/pytorch/rl).

## Key Features

*   **Python-First Design:** Built with Python for ease of use and customization.
*   **Optimized for Efficiency:** Designed for high performance to accelerate RL research.
*   **Modular & Extensible:** Easily swap, transform, or create custom components.
*   **Comprehensive Documentation:** Well-documented for quick understanding and use.
*   **Rigorously Tested:** Ensures reliability and stability.
*   **Reusable Functionals:** Provides a rich set of reusable functions for key RL components.
*   **LLM API**: Build RLHF, SFT and tool-augmented training pipelines.

## What's New: LLM API

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

*   **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
*   **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
*   **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
*   **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
*   **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
*   **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

## Core Components

*   **Environment Integration:** Seamlessly integrates with popular environments like OpenAI Gym and others, with batched and parallel execution capabilities.
*   **Data Collection:** Supports multi-process and distributed data collection for fast and efficient data gathering.
*   **Replay Buffers:** Provides efficient and generic replay buffers, also offered as wrappers around common datasets for *offline RL*.
*   **Environment Transforms:** Offers a flexible and vectorized environment transformation system to preprocess environment data.
*   **Modular Architectures:** Provides a range of pre-built architectures and models.
*   **Exploration Wrappers:** Easily switch between exploration and exploitation strategies.
*   **Loss Modules and Functional Computations:** Includes efficient loss modules and vectorized implementations for value estimation and advantage calculations.
*   **Trainer Class:** A generic trainer class with a hooking mechanism to support logging and data transformations.
*   **Model Recipes:** Provides recipes to build models that correspond to the environment being deployed.

## Writing simplified and portable RL codebase with `TensorDict`

RL algorithms are very heterogeneous, and it can be hard to recycle a codebase
across settings (e.g. from online to offline, from state-based to pixel-based 
learning).
TorchRL solves this problem through [`TensorDict`](https://github.com/pytorch/tensordict/),
a convenient data structure<sup>(1)</sup> that can be used to streamline one's
RL codebase.

*   **Simplified Code:** Enables concise RL code with a complete PPO training script in under 100 lines.
*   **Data-Driven Approach:**  Provides an elegant and reusable data handling system.

## Get Started

Explore our tutorials and examples to quickly get up and running with TorchRL.
*   [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started)

## Documentation and Knowledge Base

Access comprehensive documentation and a knowledge base to help you navigate and master TorchRL.
*   [Documentation](https://pytorch.org/rl)
*   [Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html)

## Spotlight Publications

TorchRL has been used in a variety of applications:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Installation

Follow these steps to install TorchRL:

### Create a virtual environment:

```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

Or create a conda environment where the packages will be installed.

```
conda create --name torchrl python=3.9
conda activate torchrl
```

### Install dependencies:

TorchRL offers a few pre-defined dependencies such as `"torchrl[tests]"`, `"torchrl[atari]"` etc. 

Install the **latest stable release**:

```bash
pip3 install torchrl
```

For more complex cases, refer to the original README for detailed installation instructions.

## How to Contribute

We welcome your contributions!  Please see the [CONTRIBUTING.md](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for guidelines.  If you encounter a bug, please open an issue in the repository.

## License

TorchRL is released under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.