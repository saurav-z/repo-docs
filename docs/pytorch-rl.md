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

# TorchRL: Open-Source Reinforcement Learning for PyTorch

TorchRL is a powerful, open-source library that brings cutting-edge reinforcement learning capabilities to PyTorch, making RL research and development more accessible and efficient.  Explore the code on [GitHub](https://github.com/pytorch/rl).

## Key Features

*   **Python-First Design:** Built with Python for ease of use, flexibility, and rapid prototyping.
*   **High Performance:** Optimized for efficient execution, supporting demanding RL research applications.
*   **Modular and Extensible:**  Modular architecture enables easy customization, component swapping, and the creation of new functionalities.
*   **Comprehensive Documentation:** Thorough documentation ensures users can quickly understand and utilize the library effectively.
*   **Rigorous Testing:**  Rigorously tested to ensure reliability and stability.
*   **Reusable Components:** Offers a set of highly reusable functions for cost functions, returns, and data processing.
*   **LLM API:** Complete framework for Language Model fine-tuning! 
    *   Unified LLM Wrappers with Hugging Face and vLLM support
    *   Advanced Conversation Management with the `History` class
    *   Tool integration for Python code execution, function calling and custom tool transforms.
    *   Specialized Objectives include GRPO (Group Relative Policy Optimization) and SFT loss functions.
    *   High-performance Collectors with distributed training support.
    *   Flexible environments with a transform-based architecture.

## What's New: LLM API

TorchRL introduces a comprehensive **LLM API** for post-training and fine-tuning of language models, providing everything you need for RLHF, supervised fine-tuning, and tool-augmented training.

*   🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
*   💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
*   🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
*   🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
*   ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
*   🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

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

Start your RL journey with TorchRL using our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

*   Comprehensive documentation available [here](https://pytorch.org/rl), including tutorials and API references.
*   RL Knowledge base to debug code and learn the basics of RL is available [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).
*   Check out the introductory videos [here](https://pytorch.org/rl/stable/index.html#videos)

## Key Design Principles

*   **PyTorch Ecosystem Alignment:** Follows PyTorch library structures (e.g., datasets, transforms, models) for familiarity.
*   **Minimal Dependencies:**  Requires only Python standard library, NumPy, and PyTorch.  Optional dependencies for common environment libraries and datasets.

Read the [full paper](https://arxiv.org/abs/2306.00577) for a curated description of the library.

## Spotlight Publications

TorchRL is being used in many different fields. Here are a few examples:

-   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
-   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
-   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
-   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
-   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
-   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL uses `TensorDict`, a convenient data structure to streamline RL codebases. This makes it possible to write complete PPO training scripts in less than 100 lines of code!

  <details>
    <summary>Code</summary>

  ```python
  import torch
  from tensordict.nn import TensorDictModule
  from tensordict.nn.distributions import NormalParamExtractor
  from torch import nn
  
  from torchrl.collectors import SyncDataCollector
  from torchrl.data.replay_buffers import TensorDictReplayBuffer, \
    LazyTensorStorage, SamplerWithoutReplacement
  from torchrl.envs.libs.gym import GymEnv
  from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
  from torchrl.objectives import ClipPPOLoss
  from torchrl.objectives.value import GAE
  
  env = GymEnv("Pendulum-v1") 
  model = TensorDictModule(
    nn.Sequential(
        nn.Linear(3, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 2),
        NormalParamExtractor()
    ),
    in_keys=["observation"],
    out_keys=["loc", "scale"]
  )
  critic = ValueOperator(
    nn.Sequential(
        nn.Linear(3, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    ),
    in_keys=["observation"],
  )
  actor = ProbabilisticActor(
    model,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={"low": -1.0, "high": 1.0},
    return_log_prob=True
    )
  buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(1000),
    sampler=SamplerWithoutReplacement(),
    batch_size=50,
    )
  collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=1000,
    total_frames=1_000_000,
  )
  loss_fn = ClipPPOLoss(actor, critic)
  adv_fn = GAE(value_network=critic, average_gae=True, gamma=0.99, lmbda=0.95)
  optim = torch.optim.Adam(loss_fn.parameters(), lr=2e-4)
  
  for data in collector:  # collect data
    for epoch in range(10):
        adv_fn(data)  # compute advantage
        buffer.extend(data)
        for sample in buffer:  # consume data
            loss_vals = loss_fn(sample)
            loss_val = sum(
                value for key, value in loss_vals.items() if
                key.startswith("loss")
                )
            loss_val.backward()
            optim.step()
            optim.zero_grad()
    print(f"avg reward: {data['next', 'reward'].mean().item(): 4.4f}")
  ```
  </details>

See [TensorDict tutorials](https://pytorch.github.io/tensordict/) to learn more.

## Features

*   **Environment Interface:** A common interface for environments supporting popular libraries (OpenAI Gym, etc.) and state-less execution.
*   **Data Collection:** Multiprocess and distributed data collectors.
*   **Replay Buffers:** Efficient and generic replay buffers with modularized storage.
*   **Environment Transforms:** Cross-library environment transforms for data processing.
*   **Architectures and Models:** Variety of pre-built architectures and models (e.g., actor-critic).
*   **Exploration Tools:** Exploration wrappers and modules to easily switch between exploration and exploitation strategies.
*   **Loss Modules and Advantage Computation:** Efficient loss modules and highly vectorized functional return and advantage computation.
*   **Trainer Class:** A generic trainer class to execute training loops.
*   **Recipes and LLM API**: Provides recipes to build models and a complete framework for language model fine-tuning.

## Examples, Tutorials and Demos

State-of-the-Art implementations, code examples and tutorials are available in the repository.

*   **State-of-the-Art Implementations:**  [DQN, DDPG, IQL, CQL, TD3, PPO, SAC, REDQ, Dreamer v1, Decision Transformers, CrossQ, Gail, Impala, MARL implementations.](https://github.com/pytorch/rl/blob/main/sota-implementations/)
*   **Code Examples:** [LLM API & GRPO](sota-implementations/grpo), [RLHF](examples/rlhf), [Memory-mapped replay buffers](examples/torchrl_features)
*   **Tutorials and Demos:**  Visit [the documentation](https://pytorch.org/rl/stable#tutorials) to get started.

## Citation

If you use TorchRL in your research, please cite it using the BibTeX entry provided in the original README.

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

TorchRL requires PyTorch to be installed. You can install either the latest (nightly) PyTorch release or the latest stable version of PyTorch. See [here](https://pytorch.org/get-started/locally/) for a detailed list of commands.

#### TorchRL

You can install the latest stable release by using:
```bash
pip3 install torchrl
```

The **nightly build** can be installed via
```bash
pip3 install tensordict-nightly torchrl-nightly
```

**Disclaimer**: As of today, TorchRL is roughly compatible with any pytorch version >= 2.1.
The C++ binaries of TorchRL (mainly for prioritized replay buffers) will only work with PyTorch 2.7.0 and above.
Some features (e.g., working with nested jagged tensors) may also be limited with older versions of pytorch. It is recommended to use the latest TorchRL with the latest PyTorch version unless there is a strong reason not to do so.

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

*   Report bugs and issues in this repository.
*   Ask general RL questions on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) guide for details.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.