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

# TorchRL: The PyTorch Library for Cutting-Edge Reinforcement Learning

[TorchRL](https://github.com/pytorch/rl) is an open-source library that empowers researchers and developers to build advanced reinforcement learning (RL) models with PyTorch.

## Key Features

*   **Python-First Design:** Simplifies RL development with an intuitive and flexible Python-centric approach.
*   **High Performance:** Optimized for speed, enabling efficient training and experimentation for demanding RL applications.
*   **Modular and Extensible:** Offers a flexible architecture for easy customization, allowing you to swap, transform, or create new components.
*   **Comprehensive Documentation:** Includes thorough documentation and tutorials for quick understanding and utilization.
*   **Rigorous Testing:** Ensures reliability and stability through extensive testing.
*   **Reusable Functionals:** Provides a library of reusable functions for cost functions, returns, and data processing.
*   **LLM API**: A Complete framework for language model fine-tuning.
*   **TensorDict Integration**: Write a complete PPO training script in less than 100 lines of code.

## What's New: LLM API for Language Model Fine-tuning

TorchRL introduces a powerful **LLM API** to fine-tune language models. This offers everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

*   **Unified LLM Wrappers:** Seamlessly integrate with Hugging Face models and vLLM inference engines.
*   **Conversation Management:** Advanced `History` class for multi-turn dialogue with automatic chat template detection.
*   **Tool Integration:** Built-in support for Python code execution, function calling, and custom tool transforms.
*   **Specialized Objectives:** GRPO (Group Relative Policy Optimization) and SFT loss functions optimized for language models.
*   **High-Performance Collectors:** Async data collection with distributed training support.
*   **Flexible Environments:** Transform-based architecture for reward computation, data loading, and conversation augmentation.

Explore the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started.

## Why Choose TorchRL?

*   **Alignment with PyTorch:** Follows PyTorch conventions for ease of use.
*   **Minimal Dependencies:** Uses only Python standard library, NumPy, and PyTorch, with optional dependencies for common environments and datasets.

Read the [full paper](https://arxiv.org/abs/2306.00577) for a deeper dive into TorchRL.

## Getting Started

Get up and running quickly with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

## Documentation and Resources

*   **Documentation:** Access detailed documentation, tutorials, and the API reference [here](https://pytorch.org/rl).
*   **Knowledge Base:** Find debugging tips and RL fundamentals in the [RL knowledge base](https://pytorch.org/rl/stable/reference/knowledge_base.html).
*   **Introductory Videos:**
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Applications in the Real World

TorchRL is a versatile tool for research and industry, applied in various fields:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): Reinforcement Learning for Combinatorial Optimization
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A Unified Framework for Robot Learning

## Simplify Your RL Code with TensorDict

TorchRL utilizes `TensorDict`, a data structure designed to streamline your RL code:

*   Easily reuse code across different environments, models, and algorithms.
*   Streamlines your RL codebase.

## Installation

Follow these steps to set up TorchRL.

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

## Contribute and Get Involved

*   **Report Issues:** Help improve the library by reporting any bugs you find.
*   **Ask Questions:** Get support and discuss RL topics on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).
*   **Contribute:** Join the community by forking the repository and submitting pull requests. See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for details.

## License

TorchRL is available under the [MIT License](https://github.com/pytorch/rl/blob/main/LICENSE).