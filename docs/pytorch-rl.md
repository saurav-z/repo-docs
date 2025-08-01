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

# TorchRL: Your PyTorch-Powered Toolkit for Reinforcement Learning

TorchRL is an open-source, PyTorch-based library designed to simplify and accelerate your Reinforcement Learning (RL) research and development. [Explore the full repository](https://github.com/pytorch/rl)!

## Key Features

*   **Python-First & User-Friendly:** Built with Python for ease of use, flexibility, and a familiar development experience.
*   **High Performance:** Optimized for speed to support demanding RL applications.
*   **Modular & Extensible:** Highly modular design allows for easy customization and the creation of new components.
*   **Well-Documented:** Comprehensive documentation with tutorials and API reference.
*   **Rigorously Tested:** Ensures reliability and stability.
*   **Reusable Components:** Provides a rich set of reusable functions for cost functions, returns, and data processing.
*   **LLM API** Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, 
  conversation management with automatic chat template detection, tool integration (Python execution, function calling), 
  specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning, 
  and tool-augmented training scenarios.

## What's New: LLM API 🚀

TorchRL now features a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

*   **Unified LLM Wrappers:** Seamless integration with Hugging Face models and vLLM inference engines.
*   **Conversation Management:** Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
*   **Tool Integration:** [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
*   **Specialized Objectives:** [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
*   **High-Performance Collectors:** [Async data collection](torchrl/collectors/llm/) with distributed training support
*   **Flexible Environments:** Transform-based architecture for reward computation, data loading, and conversation augmentation

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

Jumpstart your RL journey with the [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

## Documentation and Knowledge Base

*   **Comprehensive Documentation:** Find detailed information, tutorials, and the API reference [here](https://pytorch.org/rl).
*   **RL Knowledge Base:** Learn the fundamentals and debug your code with the RL knowledge base [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

## Spotlight Publications

TorchRL's versatility is demonstrated across diverse fields. Check out these publications:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
  Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Simplify RL with TensorDict

TorchRL streamlines RL development with [`TensorDict`](https://github.com/pytorch/tensordict/), a powerful data structure for managing data efficiently, leading to concise code.  A *complete PPO training script can be written in less than 100
lines of code*!

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

## Features in Detail

*   **Environments:** Unified interface, supports common libraries (OpenAI Gym, etc.), batched environments for parallel execution, and tensor specifications.  [More on Environments](https://pytorch.org/rl/stable/reference/envs.html) and [tutorials](https://pytorch.org/rl/stable/tutorials/pendulum.html).
*   **Data Collectors:** Multiprocess and distributed data collection, both synchronous and asynchronous. Explore [distributed collector examples](https://github.com/pytorch/rl/blob/main/examples/distributed/collectors).
*   **Replay Buffers:** Efficient and generic replay buffers with modular storage, also offered as wrappers around common datasets for *offline RL*.
*   **Environment Transforms:** Cross-library environment transforms executed on device in a vectorized fashion.
*   **Models and Architectures:** Ready-to-use architectures and models including actor-critic.
*   **Exploration:** Wrappers and modules for easily swapping between exploration and exploitation.
*   **Loss Modules:** Efficient loss modules and vectorized return/advantage computation.
*   **Trainer Class:** A generic trainer class for executing training loops with a hooking mechanism for logging.
*   **Recipes:** Provide recipes to build models.

## Examples, Tutorials, and Demos

Explore our collection of [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) including:

| Algorithm  | Compile Support | Modular Losses | Continuous/Discrete |
|----------------|-----------------|----------------|---------------------|
| DQN          | 1.9x            | NA             | + (via ActionDiscretizer)     |
| DDPG         | 1.87x           | +              | - (continuous)        |
| IQL          | 3.22x           | +              | +                   |
| CQL          | 2.68x           | +              | +                   |
| TD3          | 2.27x           | +              | - (continuous)        |
| TD3+BC       | untested        | +              | - (continuous)        |
| A2C          | 2.67x           | -              | +                   |
| PPO          | 2.42x           | -              | +                   |
| SAC          | 2.62x           | -              | +                   |
| REDQ         | 2.28x           | -              | - (continuous)        |
| Dreamer v1   | untested        | + (different classes) | - (continuous)        |
| Decision Transformers | untested | NA  | - (continuous) |
| CrossQ       | untested        | +              | - (continuous)        |
| Gail         | untested        | NA  | + |
| Impala         | untested  | -              | +                   |
| IQL (MARL)        | untested  | +              | +                   |
| DDPG (MARL)        | untested  | +              | - (continuous)                    |
| PPO (MARL)        | untested  | -              | +                   |
| QMIX-VDN (MARL)        | untested  | NA  | + |
| SAC (MARL)        | untested  | -              | +                   |
| RLHF | NA  | NA | NA                   |
| LLM API (GRPO) | NA  | + | NA |

and more!

**Examples:**

*   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

## Citation

If you use TorchRL, please cite our work:

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

Install the appropriate PyTorch version for your needs.  See [here](https://pytorch.org/get-started/locally/) for installation instructions.

#### Torchrl

Install the **latest stable release** by using
```bash
pip3 install torchrl
```

**Nightly builds** can be installed via:

```bash
pip3 install tensordict-nightly torchrl-nightly
```

For detailed installation instructions, troubleshooting, and optional dependencies, see the original [README](https://github.com/pytorch/rl).

## Get Involved

*   **Report Bugs:**  If you find a bug, please [raise an issue](https://github.com/pytorch/rl/issues) in this repository.
*   **Ask Questions:** For general questions, use the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).
*   **Contribute:**  We welcome contributions! See our [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) and [open contributions](https://github.com/pytorch/rl/issues/509).

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.