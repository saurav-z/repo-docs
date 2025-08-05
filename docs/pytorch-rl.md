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

TorchRL is an open-source library designed to simplify and accelerate your Reinforcement Learning (RL) research using PyTorch; explore its [GitHub repository](https://github.com/pytorch/rl) for more details.

## Key Features:

*   🐍 **Python-First Design:** Built with Python at its core for ease of use and unparalleled flexibility in your RL projects.
*   ⏱️ **High Performance:** Optimized for speed, enabling efficient training and experimentation for complex RL applications.
*   🧮 **Modular and Extensible:** A highly modular architecture lets you easily swap, transform, or create new components tailored to your specific needs.
*   📚 **Comprehensive Documentation:**  Thoroughly documented to ensure quick understanding and seamless integration into your workflows.
*   ✅ **Rigorous Testing:**  Rigorously tested to ensure stability and reliability in the face of complex challenges.
*   ⚙️ **Reusable Functionals:** Offers a rich suite of reusable functions for cost functions, return calculations, and optimized data processing.
*   🤖 **LLM API:** A complete framework for language model fine-tuning including wrappers, conversation management, tool integration, specialized objectives, and high-performance data collection.

## What's New: LLM API for Language Models

TorchRL introduces a comprehensive **LLM API** for post-training and fine-tuning of language models! This API includes:

*   🤖 **Unified LLM Wrappers:** Compatible with Hugging Face models and vLLM inference engines.
*   💬 **Conversation Management:** Advanced `History` class for multi-turn dialogue.
*   🛠️ **Tool Integration:** Built-in support for Python code execution and function calls.
*   🎯 **Specialized Objectives:** GRPO (Group Relative Policy Optimization) and SFT loss functions.
*   ⚡ **High-Performance Collectors:** Async data collection.
*   🔄 **Flexible Environments:** Transform-based architecture for reward computation.

## Get Started Quickly

Explore our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to rapidly get up to speed with TorchRL's essential features and begin your RL journey.

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

Access comprehensive documentation and an extensive knowledge base to support your RL projects:

*   [Documentation](https://pytorch.org/rl): Find tutorials, API references, and more.
*   [RL Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html): Troubleshoot your code and learn RL fundamentals.

## Real-World Applications

TorchRL is versatile and applicable across diverse domains.  Here are some examples of its usage:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): RL for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent RL
*   [BricksRL](https://arxiv.org/abs/2406.17490): Robotics and RL with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): Drone Control with RL
*   [RL4CO](https://arxiv.org/abs/2306.17100): RL for Combinatorial Optimization
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): Robot Learning Framework

## Simplify RL Code with TensorDict

Leverage the `TensorDict` data structure for streamlined and portable RL codebases.  TorchRL provides:

*   Complete PPO training scripts in under 100 lines of code.
*   An easy-to-use environment API.
*   Seamless data flow across environments, models, and algorithms.
*   Support for numerous tensor operations, including stacking, reshaping, and device transfers.

Explore the [TensorDict tutorials](https://pytorch.github.io/tensordict/) to learn more.

## Installation

Follow these steps to get started:

### 1. Set up your environment.
```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

OR

```
conda create --name torchrl python=3.9
conda activate torchrl
```

### 2. Install PyTorch.
Visit [PyTorch Installation](https://pytorch.org/get-started/locally/) for detailed instructions, choosing the version that aligns with your needs.

### 3. Install TorchRL.

```bash
pip3 install torchrl
```
See the original README for additional build and install instructions.

### 4. Install optional dependencies
```bash
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher
pip3 install "moviepy<2.0.0" # rendering
pip3 install dm_control # deepmind control suite
pip3 install "gym[atari]" "gym[accept-rom-license]" pygame # gym, atari games
pip3 install pytest pyyaml pytest-instafail  # tests
pip3 install tensorboard # tensorboard
pip3 install wandb # wandb
```
##  Contributing

We welcome your contributions! Review our detailed contribution guide [here](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).  For active contribution opportunities, see [open contributions](https://github.com/pytorch/rl/issues/509).

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.