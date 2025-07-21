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

# TorchRL: Accelerate Your Reinforcement Learning Research with PyTorch

[TorchRL](https://github.com/pytorch/rl) is an open-source library that provides powerful tools and a flexible framework for Reinforcement Learning (RL) research, built on top of PyTorch.

## Key Features

*   🐍 **Python-first:** Designed for ease of use and flexibility with Python as the primary language.
*   ⏱️ **Efficient:** Optimized for performance to support demanding RL applications.
*   🧮 **Modular, Customizable, Extensible:** Highly modular architecture for easy component swapping, transformation, and creation.
*   📚 **Well-Documented:** Thorough documentation for quick understanding and library utilization.
*   ✅ **Rigorously Tested:** Ensures reliability and stability.
*   ⚙️ **Reusable Functionals:** Provides reusable functions for cost functions, returns, and data processing.
*   **🚀 LLM API:** Comprehensive framework for LLM fine-tuning, including wrappers for Hugging Face and vLLM, conversation management, tool integration, specialized objectives (GRPO, SFT), and high-performance collectors.

## What's New: LLM API Highlights

TorchRL's new LLM API provides a complete solution for language model fine-tuning, including RLHF, supervised fine-tuning, and tool-augmented training.

*   🤖 **Unified LLM Wrappers:** Compatible with Hugging Face models and vLLM inference engines.
*   💬 **Conversation Management:** Advanced `History` class for multi-turn dialogue.
*   🛠️ **Tool Integration:** Supports Python code execution, function calling, and custom tool transforms.
*   🎯 **Specialized Objectives:** Includes GRPO and SFT loss functions optimized for language models.
*   ⚡ **High-Performance Collectors:** Async data collection with distributed training support.
*   🔄 **Flexible Environments:** Transform-based architecture for reward computation and conversation augmentation.

Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and the [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

<details>
  <summary>Quick LLM API Example</summary>

```python
from torchrl.envs.llm import ChatEnv
from torchrl.modules.llm import TransformersWrapper
from torchrl.objectives.llm import GRPOLoss
from torchrl.collectors.llm import LLMCollector

# Create environment with Python tool execution
env = ChatEnv(
    tokenizer=tokenizer,
    system_prompt="You are an assistant that can execute Python code.",
    batch_size=[1]
).append_transform(PythonInterpreter())

# Wrap your language model
llm = TransformersWrapper(
    model=model,
    tokenizer=tokenizer,
    input_mode="history"
)

# Set up GRPO training
loss_fn = GRPOLoss(llm, critic, gamma=0.99)
collector = LLMCollector(env, llm, frames_per_batch=100)

# Training loop
for data in collector:
    loss = loss_fn(data)
    loss.backward()
    optimizer.step()
```

</details>

## Why Use TorchRL?

*   **Aligns with PyTorch Ecosystem:** Follows the structure and conventions of other popular PyTorch libraries.
*   **Minimal Dependencies:** Only requires Python standard library, NumPy, and PyTorch.
*   **Simplifies RL Codebases:** Leverages `TensorDict` for streamlined, portable code, reducing complexity.

## Getting Started

Explore our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly learn the library's basic features.

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

*   [Documentation](https://pytorch.org/rl): Comprehensive documentation, tutorials, and API reference.
*   [RL Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html): Helps you debug your code and learn RL fundamentals.

## Spotlight Publications

TorchRL is versatile and used across various domains:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified RL Code with TensorDict

TorchRL uses `TensorDict`, a data structure, to make RL codebases more streamlined and portable.

## Installation

### Create a Virtual Environment:

```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

Or Create a Conda Environment:

```
conda create --name torchrl python=3.9
conda activate torchrl
```

### Install Dependencies:

#### PyTorch

*   Install the latest PyTorch release or the latest nightly version.  See [here](https://pytorch.org/get-started/locally/) for installation commands.

#### TorchRL

*   Install the **latest stable release**:

    ```bash
    pip3 install torchrl
    ```

    This works on Linux, Windows 10, and macOS (Metal). For Windows 11, build locally:

    ```bash
    pip3 install git+https://github.com/pytorch/rl@v0.8.1 # v0.8.1
    # OR build locally
    git clone https://github.com/pytorch/tensordict
    git clone https://github.com/pytorch/rl
    pip install -e tensordict
    pip install -e rl
    ```

*   Install the **nightly build**:

    ```bash
    pip3 install tensordict-nightly torchrl-nightly
    ```
    (Linux only. Requires nightly PyTorch).

**Optional Dependencies**
```bash
# diverse
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher

# rendering
pip3 install "moviepy<2.0.0"

# deepmind control suite
pip3 install dm_control

# gym, atari games
pip3 install "gym[atari]" "gym[accept-rom-license]" pygame

# tests
pip3 install pytest pyyaml pytest-instafail

# tensorboard
pip3 install tensorboard

# wandb
pip3 install wandb
```

For versioning issues, see [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md).

## Contributing

Welcome internal collaborations!  Fork, submit issues and PRs.  See the [detailed contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for information.
[Open contributions](https://github.com/pytorch/rl/issues/509)

## License
TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.