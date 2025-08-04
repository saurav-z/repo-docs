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

TorchRL is an open-source library that empowers researchers and developers with a comprehensive suite of tools for building and deploying reinforcement learning (RL) agents in PyTorch.  [Explore the TorchRL repository](https://github.com/pytorch/rl) to get started!

## Key Features

*   **Python-First Design:** Prioritizes Python for ease of use, flexibility, and rapid prototyping.
*   **Efficient Implementation:** Optimized for performance, enabling demanding RL research and real-world applications.
*   **Modular Architecture:**  Offers a highly modular and customizable design, facilitating component swapping, transformation, and extension.
*   **Comprehensive Documentation:** Includes thorough documentation, tutorials, and API references to ensure ease of use and understanding.
*   **Rigorous Testing:**  Maintains high reliability and stability through extensive testing.
*   **Reusable Functionals:** Provides a rich collection of reusable functions for cost functions, returns, and data processing.
*   **LLM API:** Complete framework for language model fine-tuning, including unified wrappers, conversation management, tool integration, specialized objectives, and high-performance collectors.

## What's New: LLM API 🚀

TorchRL now includes an **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

-   🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines.
-   💬 **Conversation Management**: Advanced `History` class for multi-turn dialogue with automatic chat template detection
-   🛠️ **Tool Integration**: Built-in support for Python code execution, function calling, and custom tool transforms
-   🎯 **Specialized Objectives**: `GRPO` (Group Relative Policy Optimization) and `SFT` loss functions optimized for language models
-   ⚡ **High-Performance Collectors**: Async data collection with distributed training support
-   🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

## Core Design Principles

*   **PyTorch Ecosystem Alignment:** Adheres to the structure and conventions of popular PyTorch libraries, e.g., datasets, transforms, and data utilities.
*   **Minimal Dependencies:**  Relies primarily on the Python standard library, NumPy, and PyTorch. Optional dependencies for environment libraries (e.g., OpenAI Gym) and datasets (e.g., D4RL, OpenX).

For a deeper understanding of the library's design, refer to the [full paper](https://arxiv.org/abs/2306.00577).

## Getting Started

Start your RL journey with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

## Documentation and Knowledge Base

Access the comprehensive TorchRL documentation [here](https://pytorch.org/rl), featuring tutorials and the API reference.  Explore the RL knowledge base to learn the basics of RL and debug your code [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

## Spotlight Publications

TorchRL is a versatile tool applicable across many domains. Here are a few examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Simplified RL Codebases with TensorDict

TorchRL leverages `TensorDict` to streamline RL codebase development. This data structure simplifies the creation of reusable and portable code across settings, such as online and offline RL, and state-based and pixel-based learning. You can write a complete PPO training script in under 100 lines of code!

Learn more about `TensorDict` and its capabilities [here](https://github.com/pytorch/tensordict/).

## Features in Detail

*   **Environment Interface:**  A standardized interface for environments, including common libraries (OpenAI Gym, DeepMind Control Lab) and state-less execution.  Offers batched environments for parallel execution, and PyTorch-first tensor specifications.
*   **Data Collectors:**  Supports multi-process and distributed data collection, both synchronously and asynchronously.
*   **Replay Buffers:**  Efficient and generic replay buffers with modularized storage, including wrappers for offline RL datasets.
*   **Environment Transforms:** Cross-library environment transforms, executed on device, used to preprocess and prepare environment data for the agent.
*   **Tools for Distributed Learning:** Includes tools like memory-mapped tensors.
*   **Architectures and Models:** Provides various architectures and models, including actor-critic designs.
*   **Exploration Wrappers and Modules:** Simplifies the transition between exploration and exploitation strategies.
*   **Loss Modules and Functional Computations:** Offers efficient loss modules and highly vectorized functional return and advantage computation.
*   **Trainer Class:**  Includes a generic trainer class that executes the training loop, supports logging, and data transformation.
*   **Model Recipes:**  Offers recipes for building models tailored to specific environments.
*   **LLM API:**  Complete LLM framework (see above).

## Examples, Tutorials, and Demos

Explore the [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) for practical guidance:

*   DQN
*   DDPG
*   IQL
*   CQL
*   TD3
*   TD3+BC
*   A2C
*   PPO
*   SAC
*   REDQ
*   Dreamer v1
*   Decision Transformers
*   CrossQ
*   Gail
*   Impala
*   IQL (MARL)
*   DDPG (MARL)
*   PPO (MARL)
*   QMIX-VDN (MARL)
*   SAC (MARL)
*   RLHF
*   LLM API (GRPO)

and many more!

Find code examples in the [examples/](https://github.com/pytorch/rl/blob/main/examples/) directory. Explore training scripts, including:

*   LLM API & GRPO
*   RLHF
*   Memory-mapped replay buffers

See the [examples directory](https://github.com/pytorch/rl/blob/main/examples/) for more details.  Also, check out the [tutorials and demos](https://pytorch.org/rl/stable#tutorials) for hands-on learning.

## Citation

If you are using TorchRL, please cite this work:

```bibtex
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

### Create a new virtual environment:

```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

Or create a conda environment:

```bash
conda create --name torchrl python=3.9
conda activate torchrl
```

### Install dependencies:

#### PyTorch

Install the appropriate PyTorch version [here](https://pytorch.org/get-started/locally/).

#### TorchRL

```bash
pip3 install torchrl
```

**Alternative installation methods and considerations are detailed in the original README.**

**Optional Dependencies**: Install additional libraries based on your use case: (e.g., rendering, DeepMind Control Suite, etc. - see original README for a complete list).

Refer to the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md) if you encounter errors.

## Support and Contribution

*   **Bug Reports:**  Raise issues for any bugs found in the library.
*   **Questions:**  Post questions on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).
*   **Contributions:**  Contributions are welcome!  Review the detailed contribution guide [here](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).

## Disclaimer and License

TorchRL is released as a PyTorch beta feature. BC-breaking changes may occur.
TorchRL is licensed under the [MIT License](https://github.com/pytorch/rl/blob/main/LICENSE).