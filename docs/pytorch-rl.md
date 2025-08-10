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

TorchRL is an open-source library designed to accelerate your Reinforcement Learning (RL) research, offering a modular, efficient, and user-friendly environment for developing and deploying RL algorithms.  Find out more and contribute at the original [repository](https://github.com/pytorch/rl).

## Key Features

*   🐍 **Python-first:** Designed with Python for ease of use and flexibility.
*   ⏱️ **Efficient:** Optimized for performance to support demanding RL research applications.
*   🧮 **Modular, Customizable, Extensible:** Highly modular architecture allows for easy swapping, transformation, or creation of new components.
*   📚 **Well-Documented:** Thorough documentation ensures users can quickly understand and utilize the library.
*   ✅ **Extensively Tested:** Rigorously tested to ensure reliability and stability.
*   ⚙️ **Reusable Functionals:** Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   🧠 **LLM API**: Integrated framework for language model fine-tuning, including RLHF, with Hugging Face and vLLM wrappers, conversation management, and specialized objectives (GRPO, SFT).

## Core Components

*   **Environments:** Unified interface supporting common libraries (OpenAI Gym, DeepMind Control, etc.) and state-less execution. Includes batched environments for parallel execution and PyTorch-first tensor-specification classes.
*   **Data Collectors:** Multiprocess and distributed data collectors, working synchronously or asynchronously.
*   **Replay Buffers:** Efficient and generic replay buffers with modularized storage, including wrappers for offline RL datasets (D4RL).
*   **Environment Transforms:** Cross-library environment transforms executed on device and in a vectorized fashion.
*   **Models and Architectures:** Various architectures and models, including actor-critic implementations.
*   **Exploration Wrappers and Modules:** Easily switch between exploration and exploitation strategies.
*   **Loss Modules and Functionals:** Efficient loss modules and vectorized functional return/advantage computation.
*   **Trainer Class:** A generic trainer class supporting logging and data transformation hooks.
*   **Recipes:** Models tailored for the environment being deployed.

## 🚀 What's New: LLM API

The TorchRL library now features a comprehensive **LLM API** to facilitate the fine-tuning of language models. This includes a variety of modules for RLHF, supervised fine-tuning, and tool-augmented training. Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

## Getting Started

Check out the [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly learn the basics of the library!

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

*   [Documentation](https://pytorch.org/rl): Tutorials and API reference.
*   [RL Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html): Resources to debug your code or learn the basics of RL.

## Spotlight Publications

TorchRL is versatile and can be used across many different fields, here are a few examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL leverages `TensorDict`, a data structure that simplifies RL code by streamlining your codebase and providing a convenient tool for writing Portable RL code.
With this tool, one can write a complete PPO training script in less than 100 lines of code!

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

## Examples, Tutorials, and Demos

Explore our [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/)! Code examples and training scripts are also available.

## Citation

If you're using TorchRL, please refer to the [citation](https://github.com/pytorch/rl#citation) in the original README.

## Installation

Follow the [installation instructions](https://github.com/pytorch/rl#installation) in the original README.

## Asking a Question

*   Report bugs in the [issue tracker](https://github.com/pytorch/rl/issues).
*   Ask general questions on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome! See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for details.