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

# TorchRL: The Open-Source Reinforcement Learning Library for PyTorch

**Supercharge your RL research with TorchRL, a powerful and flexible library built for PyTorch.**  [Explore the original repo.](https://github.com/pytorch/rl)

## Key Features

*   **🚀 LLM API**: Complete framework for language model fine-tuning, including:
    *   Unified wrappers for Hugging Face and vLLM backends.
    *   Advanced conversation management.
    *   Built-in tool integration (Python execution, function calling, and custom tool transforms).
    *   Specialized objectives (GRPO, SFT).
    *   High-performance async collectors.
*   🐍 **Python-first:** Designed for ease of use and flexibility with Python as the primary language.
*   ⏱️ **Efficient:** Optimized for performance to handle demanding RL research applications.
*   🧮 **Modular, Customizable, Extensible:** Highly modular architecture for easy swapping, modification, and creation of new components.
*   📚 **Well-Documented:** Comprehensive documentation to facilitate quick understanding and usage.
*   ✅ **Rigorously Tested:** Ensures reliability and stability.
*   ⚙️ **Reusable Functionals:** Provides a set of reusable functions for cost functions, returns, and data processing.

## Core Strengths

*   **Aligned with PyTorch Ecosystem:** Follows the structure and conventions of popular PyTorch libraries for seamless integration.
*   **Minimal Dependencies:** Only requires Python standard library, NumPy, and PyTorch, with optional dependencies for common environment libraries.
*   **TensorDict Integration:** Leverage the flexibility of [TensorDict](https://pytorch.github.io/tensordict/) for simplified and portable RL codebase.

## LLM API - Get Started with Large Language Model Fine-tuning

TorchRL's LLM API provides a comprehensive suite of tools for post-training and fine-tuning language models, making it easy to build RLHF, supervised fine-tuning, and tool-augmented training applications.

Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

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

## Get Started

Explore our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly learn the library's features!

## Documentation & Knowledge Base

Comprehensive documentation is available [here](https://pytorch.org/rl) with tutorials and an API reference.  Also, check out the RL knowledge base [here](https://pytorch.org/rl/stable/reference/knowledge_base.html) to troubleshoot your code and learn RL basics.

## Spotlight Publications

TorchRL has been used in a variety of fields:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL leverages [TensorDict](https://github.com/pytorch/tensordict/), a data structure for efficient RL codebase.  This facilitates code reuse across different settings (e.g., online to offline) and allows for a complete PPO training script with a few lines of code!

Learn more about `TensorDict` with [TensorDict tutorials](https://pytorch.github.io/tensordict/).

## Examples, Tutorials, and Demos

Find [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/), code examples, training scripts, and demos, including:

*   [LLM API & GRPO](sota-implementations/grpo)
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

Consult the [examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) directory for configuration settings.

## Installation

Detailed installation instructions are provided to ensure you can quickly set up your environment.

### Create a New Virtual Environment:
```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```
### Or Create a Conda Environment
```
conda create --name torchrl python=3.9
conda activate torchrl
```

### Install Dependencies:
```bash
pip3 install torchrl
```
## Contributing

Contributions are welcome! See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for details. Open contributions can be found [here](https://github.com/pytorch/rl/issues/509).