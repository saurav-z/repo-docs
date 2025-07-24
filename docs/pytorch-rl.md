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

# TorchRL: Your Gateway to Cutting-Edge Reinforcement Learning with PyTorch

TorchRL is a powerful and flexible open-source library designed to empower researchers and developers in the field of Reinforcement Learning (RL), built on the robust PyTorch framework.  [Explore the original repository on GitHub](https://github.com/pytorch/rl) for more details.

## Key Features

*   **Python-First Design:** Enjoy an intuitive, Python-first interface for ease of use and flexibility.
*   **High Performance:** Optimized for speed, supporting demanding RL research applications.
*   **Modular Architecture:** Build and customize RL systems with a modular architecture, allowing for swapping, transforming, and creating new components.
*   **Comprehensive Documentation:** Get up and running quickly with clear documentation and tutorials.
*   **Extensive Testing:** Benefit from rigorously tested code, ensuring reliability and stability.
*   **Reusable Functionals:** Leverage a set of reusable functions for cost calculations, returns, and data processing.
*   **LLM API**: Complete framework for language model fine-tuning.

## What's New: LLM API

TorchRL now features a comprehensive LLM API for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

*   🤖 **Unified LLM Wrappers:** Seamless integration with Hugging Face models and vLLM inference engines.
*   💬 **Conversation Management:** Advanced `History` class for multi-turn dialogue.
*   🛠️ **Tool Integration:** Built-in support for Python code execution, function calling, and custom tool transforms.
*   🎯 **Specialized Objectives:** GRPO (Group Relative Policy Optimization) and SFT (Supervised Fine-tuning) loss functions.
*   ⚡ **High-Performance Collectors:** Async data collection with distributed training support.
*   🔄 **Flexible Environments:** Transform-based architecture for reward computation, data loading, and conversation augmentation.

## Getting Started

Check out the [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to learn the basics of TorchRL quickly.

## Key Advantages

*   **Alignment with PyTorch Ecosystem:** Designed to align with popular PyTorch library structures.
*   **Minimal Dependencies:** Only requires Python, NumPy, and PyTorch, with optional dependencies for common environment libraries.
*   **TensorDict Integration:** Simplifies RL codebases with the TensorDict data structure.

## Writing simplified and portable RL codebase with `TensorDict`

TorchRL solves this problem through [`TensorDict`](https://github.com/pytorch/tensordict/),
a convenient data structure<sup>(1)</sup> that can be used to streamline one's
RL codebase.
With this tool, one can write a *complete PPO training script in less than 100
lines of code*!

## Documentation

*   [TorchRL Documentation](https://pytorch.org/rl/)
*   [RL Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html)

## Spotlight Publications

TorchRL has been applied in diverse fields:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Generative Chemical Agents
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): Robotics and Reinforcement Learning with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): Combinatorial Optimization
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): Robot Learning

## Installation

### Create a new virtual environment:
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

#### PyTorch

Depending on the use of torchrl that you want to make, you may want to 
install the latest (nightly) PyTorch release or the latest stable version of PyTorch.
See [here](https://pytorch.org/get-started/locally/) for a detailed list of commands, 
including `pip3` or other special installation instructions.

TorchRL offers a few pre-defined dependencies such as `"torchrl[tests]"`, `"torchrl[atari]"` etc. 

#### Torchrl

You can install the **latest stable release** by using
```bash
pip3 install torchrl
```
This should work on linux (including AArch64 machines), Windows 10 and OsX (Metal chips only).
On certain Windows machines (Windows 11), one should build the library locally.
This can be done in two ways:

```bash
# Install and build locally v0.8.1 of the library without cloning
pip3 install git+https://github.com/pytorch/rl@v0.8.1
# Clone the library and build it locally
git clone https://github.com/pytorch/tensordict
git clone https://github.com/pytorch/rl
pip install -e tensordict
pip install -e rl
```

Note that tensordict local build requires `cmake` to be installed via [homebrew](https://brew.sh/) (MacOS) or another package manager
such as `apt`, `apt-get`, `conda` or `yum` but NOT `pip`, as well as `pip install "pybind11[global]"`.   

One can also build the wheels to distribute to co-workers using
```bash
python setup.py bdist_wheel
```
Your wheels will be stored there `./dist/torchrl<name>.whl` and installable via
```bash
pip install torchrl<name>.whl
```

The **nightly build** can be installed via
```bash
pip3 install tensordict-nightly torchrl-nightly
```
which we currently only ship for Linux machines.
Importantly, the nightly builds require the nightly builds of PyTorch too.
Also, a local build of torchrl with the nightly build of tensordict may fail - install both nightlies or both local builds but do not mix them.


**Disclaimer**: As of today, TorchRL is roughly compatible with any pytorch version >= 2.1 and installing it will not
directly require a newer version of pytorch to be installed. Indirectly though, tensordict still requires the latest
PyTorch to be installed and we are working hard to loosen that requirement. 
The C++ binaries of TorchRL (mainly for prioritized replay buffers) will only work with PyTorch 2.7.0 and above.
Some features (e.g., working with nested jagged tensors) may also
be limited with older versions of pytorch. It is recommended to use the latest TorchRL with the latest PyTorch version
unless there is a strong reason not to do so.

**Optional dependencies**

The following libraries can be installed depending on the usage one wants to
make of torchrl:
```
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

Versioning issues can cause error message of the type ```undefined symbol```
and such. For these, refer to the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md)
for a complete explanation and proposed workarounds.

## Asking a Question

If you find a bug, please submit an issue. For general RL questions in PyTorch, use the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome! See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) and the list of open contributions [here](https://github.com/pytorch/rl/issues/509).  Install [pre-commit hooks](https://pre-commit.com/) to check for linting issues before committing your code.

## Disclaimer

This is a PyTorch beta feature. BC-breaking changes are possible but will be introduced with a deprecation warranty after a few release cycles.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.