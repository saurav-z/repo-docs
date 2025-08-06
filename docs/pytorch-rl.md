<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-12-orange.svg?style=flat-square)](https://github.com/pytorch/rl/graphs/contributors)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

# TorchRL: Your Gateway to Reinforcement Learning with PyTorch

**TorchRL is an open-source library built on PyTorch, providing a modular and efficient framework for developing and experimenting with reinforcement learning (RL) algorithms.** [Explore the TorchRL Repository](https://github.com/pytorch/rl).

## Key Features:

*   🐍 **Python-First**: Designed for ease of use and flexibility with Python as the primary language.
*   ⏱️ **Efficient**: Optimized for performance to support demanding RL research applications.
*   🧮 **Modular & Customizable**: Highly modular architecture allowing easy swapping, transforming, or creating new components.
*   📚 **Well-Documented**: Thorough documentation ensuring users can quickly understand and utilize the library.
*   ✅ **Rigorously Tested**: Tested to ensure reliability and stability.
*   ⚙️ **Reusable Functionals**: Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   🤖 **LLM API**: Complete framework for Language Model Fine-tuning, enabling RLHF, supervised fine-tuning, and tool-augmented training.

## Highlights:

*   **New LLM API**: Harness the power of language models with our comprehensive API, featuring unified wrappers, advanced conversation management, tool integration, specialized objectives, and high-performance collectors. Get started with our [LLM API Documentation](https://pytorch.org/rl/main/reference/llms.html) and explore the [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo).

*   **TensorDict Integration**: Simplify your RL code with `TensorDict`, a powerful data structure enabling reusable components across environments, models, and algorithms.

*   **Modular Design**:  TorchRL aligns with the PyTorch ecosystem, minimizing dependencies (PyTorch, NumPy) and providing a flexible environment.

## Get Started:

Explore the basics with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

## Documentation and Knowledge Base:

*   [Comprehensive Documentation](https://pytorch.org/rl) with tutorials and API reference.
*   [RL Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html) to debug code and learn RL fundamentals.
*   Introductory videos are available on the [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl), [PyTorch day 2022](https://youtu.be/cIKMhZoykEE) and [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications:

TorchRL is a versatile tool for many fields. Check out the below publications:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified RL Codebases with `TensorDict`:

TorchRL leverages `TensorDict` for a streamlined RL codebase. This data structure simplifies coding. See the [TensorDict tutorials](https://pytorch.github.io/tensordict/) to learn more!

## Examples, Tutorials, and Demos:

Find State-of-the-Art implementations and example code:

*   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

and much more!
See the [examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) directory for more information.
Check out the [tutorials and demos](https://pytorch.org/rl/stable#tutorials)

## Installation:

Follow these instructions to install TorchRL.

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
pip install build
python -m build --wheel
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

## Asking a Question:

*   Report bugs by raising an issue in the [repository](https://github.com/pytorch/rl).
*   For general RL questions, post on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing:

Contributions are welcome! Check out the [detailed contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) to contribute to TorchRL. A list of open contributions can be found [here](https://github.com/pytorch/rl/issues/509).
Contributors are recommended to install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`). pre-commit will check for linting related issues when the code is committed locally. You can disable th check by appending `-n` to your commit command: `git commit -m <commit message> -n`

## Disclaimer:

TorchRL is a PyTorch beta feature. Breaking changes are possible but will be introduced with a deprecation warning.

## License:

TorchRL is licensed under the MIT License. See the [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.