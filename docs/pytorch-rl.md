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

# TorchRL: Unleash the Power of Reinforcement Learning with PyTorch

[**TorchRL**](https://github.com/pytorch/rl) is an open-source PyTorch library designed for researchers and practitioners to develop and deploy reinforcement learning (RL) algorithms efficiently and flexibly.

## Key Features

*   **Python-first Design:**  Embraces Python for ease of use, flexibility, and a familiar coding experience.
*   **Optimized for Performance:** Designed for efficient execution to support demanding RL research applications.
*   **Modular and Extensible:** Allows for easy swapping, transforming, and creating of new components.
*   **Comprehensive Documentation:** Provides thorough documentation for quick understanding and library utilization.
*   **Rigorously Tested:** Includes robust testing to ensure reliability and stability.
*   **Reusable Functionals:** Offers a rich set of reusable functions for cost functions, returns, and data processing.
*   **LLM API:** Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, conversation management with automatic chat template detection, tool integration (Python execution, function calling), specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning, and tool-augmented training scenarios.

## Core Principles

*   **PyTorch Ecosystem Alignment:** Follows the structure and conventions of popular PyTorch libraries (e.g., datasets, transforms, models, data utilities).
*   **Minimal Dependencies:** Only requires Python standard library, NumPy, and PyTorch; with optional dependencies for common environment libraries and datasets.

## What's New: LLM API

TorchRL now features a comprehensive LLM API:

*   **Unified LLM Wrappers:** Seamless integration with Hugging Face models and vLLM inference engines.
*   **Conversation Management:** Advanced `History` class for multi-turn dialogue.
*   **Tool Integration:** Built-in support for Python code execution and function calling.
*   **Specialized Objectives:** GRPO and SFT loss functions for language models.
*   **High-Performance Collectors:** Async data collection with distributed training support.
*   **Flexible Environments:** Transform-based architecture for reward computation, data loading, and conversation augmentation.

## Getting Started

Quickly get up and running with TorchRL using our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

## Key Components

*   **Environments:**  Common interface supporting libraries like OpenAI Gym.
*   **Data Collectors:**  Multiprocess and distributed data collection.
*   **Replay Buffers:**  Efficient and generic replay buffers.
*   **Environment Transforms:** On-device, vectorized data processing.
*   **Models and Architectures:**  Variety of models and architectures.
*   **Loss Modules:** Efficient loss modules.
*   **Trainers:** Generic trainer class for streamlined training.

## Featured Publications

TorchRL has been used in various research areas:

*   ACEGEN: Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   BenchMARL: Benchmarking Multi-Agent Reinforcement Learning
*   BricksRL: A Platform for Democratizing Robotics and Reinforcement Learning
*   OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   Robohive: A unified framework for robot learning

## Simplify RL Code with TensorDict

TorchRL utilizes [`TensorDict`](https://github.com/pytorch/tensordict/) to simplify RL codebases.  This enables you to write a complete PPO training script in under 100 lines of code.

## Documentation and Knowledge Base

Find in-depth information and tutorials in the [TorchRL documentation](https://pytorch.org/rl).  Also, explore our RL knowledge base for debugging tips and basic RL concepts [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

## Examples, Tutorials, and Demos

Explore [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) and [Code examples](examples/) for various algorithms and tasks, including LLM API, RLHF, and memory-mapped replay buffers.

## Citation

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

*   Create a virtual environment (recommended) or use Conda.
*   Install PyTorch (refer to [PyTorch installation instructions](https://pytorch.org/get-started/locally/)).
*   Install TorchRL via `pip3 install torchrl` (or build from source, see original README for options).

## Contributing

We welcome contributions! Review our [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) and [open contributions](https://github.com/pytorch/rl/issues/509).

## License

TorchRL is licensed under the MIT License.  See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.