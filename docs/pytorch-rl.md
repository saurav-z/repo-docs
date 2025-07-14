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

# TorchRL: Build Cutting-Edge Reinforcement Learning Solutions with PyTorch

TorchRL is an open-source library for Reinforcement Learning (RL) built on PyTorch, offering a modular, efficient, and user-friendly toolkit for both research and production. ([See the original repo](https://github.com/pytorch/rl)).

## Key Features

*   **Modular Design:** Easily swap, transform, or create new components to customize RL algorithms.
*   **Python-First Approach:** Designed with Python as the primary language for ease of use and flexibility.
*   **Efficient Execution:** Optimized for high performance to support demanding RL research.
*   **Comprehensive Documentation:** Thoroughly documented with tutorials and an API reference.
*   **Extensive Testing:** Rigorously tested to ensure reliability and stability.
*   **Reusable Functionals:** Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   **LLM API:** A complete framework for language model fine-tuning, including RLHF, supervised fine-tuning, and tool-augmented training.

## 🚀 What's New

### LLM API - Complete Framework for Language Model Fine-tuning

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

- 🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
- 💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
- 🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
- 🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
- ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
- 🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

The LLM API follows TorchRL's modular design principles, allowing you to mix and match components for your specific use case. Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

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


## Getting Started

Explore our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly ramp up with the core features of TorchRL.

## Documentation and Knowledge Base

Access detailed documentation, tutorials, and the API reference [here](https://pytorch.org/rl/).  Explore the RL knowledge base to debug code and learn RL fundamentals [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

*   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
*   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
*   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is utilized across a range of domains:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): RL for Generative Chemical Agents
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent RL
*   [BricksRL](https://arxiv.org/abs/2406.17490): Democratizing Robotics and RL with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): RL in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): RL for Combinatorial Optimization
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): Unified Robot Learning Framework

## Writing Simplified RL Code with TensorDict

TorchRL simplifies RL codebase development with `TensorDict`, a flexible data structure designed to streamline your workflow. Using this, you can code a complete PPO training script in less than 100 lines. See the details in the original documentation.

## Features

*   **Environments:** Unified interface for environments (OpenAI Gym, DeepMind, etc.).
*   **Data Collection:** Multiprocess and distributed data collectors.
*   **Replay Buffers:** Efficient and generic replay buffers with modular storage.
*   **Environment Transforms:** Cross-library environment transforms executed on device.
*   **Model Architectures:** Various architectures and models (actor-critic).
*   **Exploration Wrappers:** Easily swap between exploration and exploitation strategies.
*   **Loss Modules:** Efficient loss modules and functional return and advantage calculation.
*   **Trainer Class:** Generic trainer class for executing training loops.
*   **LLM API:** A complete framework for language model fine-tuning, including RLHF, supervised fine-tuning, and tool-augmented training.

## Examples, Tutorials, and Demos

Explore the [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) and code examples for practical application:

*   **DQN**
*   **DDPG**
*   **IQL**
*   **CQL**
*   **TD3**
*   **TD3+BC**
*   **A2C**
*   **PPO**
*   **SAC**
*   **REDQ**
*   **Dreamer v1**
*   **Decision Transformers**
*   **CrossQ**
*   **Gail**
*   **Impala**
*   **IQL (MARL)**
*   **DDPG (MARL)**
*   **PPO (MARL)**
*   **QMIX-VDN (MARL)**
*   **SAC (MARL)**
*   **RLHF**
*   **LLM API (GRPO)**

We also provide [tutorials and demos](https://pytorch.org/rl/stable#tutorials) for a deeper understanding.

## Citation

If you're using TorchRL, please refer to this BibTeX entry to cite this work:
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

Follow the installation instructions in the original documentation.

## Asking a Question

For bugs, please raise an issue in this repo.

For generic RL questions, use the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome! See the [CONTRIBUTING.md](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) guide. A list of open contributions can be found [here](https://github.com/pytorch/rl/issues/509).