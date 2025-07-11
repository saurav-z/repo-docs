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

**TorchRL** is a comprehensive and flexible open-source library for Reinforcement Learning (RL) built on PyTorch, offering powerful tools and a modular design to streamline your RL projects.  [Explore the original repository](https://github.com/pytorch/rl).

## Key Features

*   **Python-First & User-Friendly**: Designed with Python for ease of use and flexibility, aligning with the PyTorch ecosystem.
*   **Efficient & High-Performance**: Optimized for speed, enabling efficient training for demanding RL applications.
*   **Modular & Customizable**:  Highly modular architecture allows for easy swapping, transformation, or creation of new components, promoting extensibility.
*   **Well-Documented**:  Comprehensive documentation with tutorials and API reference for rapid learning and implementation.
*   **Reliable & Tested**: Rigorously tested to ensure reliability and stability.
*   **Reusable Components**: Provides a suite of reusable functions for cost functions, returns, and data processing.
*   **LLM API**: A complete framework for language model fine-tuning.

    *   🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines.
    *   💬 **Conversation Management**: Advanced `History` class for multi-turn dialogue with automatic chat template detection.
    *   🛠️ **Tool Integration**: Built-in support for Python code execution, function calling, and custom tool transforms.
    *   🎯 **Specialized Objectives**: GRPO (Group Relative Policy Optimization) and SFT loss functions optimized for language models.
    *   ⚡ **High-Performance Collectors**: Async data collection with distributed training support.
    *   🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation.

## Core Components

*   **TensorDict**: Simplify and streamline your RL codebase with TensorDict, a powerful data structure for RL tasks.
*   **Environment Integration**: A unified interface for environments, supporting popular libraries like OpenAI Gym and DeepMind Control.
*   **Data Collection**: Multi-process and distributed data collectors for efficient data gathering.
*   **Replay Buffers**: Efficient and generic replay buffers with modularized storage for offline RL.
*   **Environment Transforms**: Cross-library environment transforms executed on device in a vectorized fashion.
*   **Models & Architectures**: A variety of architectures and exploration wrappers to easily swap between exploration and exploitation.
*   **Loss Modules**: Efficient loss modules and highly vectorized return and advantage computation.
*   **Training**: A generic trainer class with a hooking mechanism, supporting logging and data transformation.

## Getting Started

Explore the [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly ramp up with the basic features of the library!

## Documentation and Knowledge Base

*   Access comprehensive documentation, tutorials, and API references [here](https://pytorch.org/rl).
*   Explore the RL knowledge base to debug your code and understand RL fundamentals [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

## Spotlight Publications

TorchRL is used across various fields. See some examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
  for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
  Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

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

If you spot a bug in the library, please raise an issue in this repo.

For generic RL questions, post them on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome! Check the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) and open contributions [here](https://github.com/pytorch/rl/issues/509).

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.