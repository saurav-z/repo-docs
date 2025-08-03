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

# TorchRL:  Open-Source Reinforcement Learning for PyTorch

**Supercharge your RL research and development with TorchRL, a PyTorch-first library for building flexible, efficient, and modular reinforcement learning systems.**  [Explore the TorchRL Repository](https://github.com/pytorch/rl).

## Key Features

*   🐍 **Python-first Design:** Experience the flexibility and ease of use that Python offers.
*   ⏱️ **High-Performance:** Benefit from optimized performance for demanding RL research.
*   🧮 **Modular Architecture:** Easily swap, transform, or create new components with a highly modular structure.
*   📚 **Comprehensive Documentation:** Understand and utilize the library quickly with thorough documentation.
*   ✅ **Rigorous Testing:** Ensure reliability and stability with extensive testing.
*   ⚙️ **Reusable Functionals:** Leverage a set of highly reusable functions for cost calculations, returns, and data processing.
*   🤖 **LLM API** - Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, 
  conversation management with automatic chat template detection, tool integration (Python execution, function calling), 
  specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning, 
  and tool-augmented training scenarios.

## What's New: LLM API

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

- 🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
- 💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
- 🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
- 🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
- ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
- 🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

The LLM API follows TorchRL's modular design principles, allowing you to mix and match components for your specific use case. Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

## Getting Started

Quickly ramp up with the basic features of the library with the  [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started)!

## Documentation and Knowledge Base

Access the complete [TorchRL documentation](https://pytorch.org/rl) including tutorials and the API reference.  Also, explore the  [RL knowledge base](https://pytorch.org/rl/stable/reference/knowledge_base.html) to aid in debugging your code and understanding the fundamentals of RL.

*   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
*   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
*   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is used across various domains. Here are a few examples:

-   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
-   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
-   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
-   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
-   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
-   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified RL Code with TensorDict

TorchRL leverages `TensorDict` to streamline RL codebases, enabling you to write complete PPO training scripts in under 100 lines of code. Learn more with the  [`TensorDict` documentation](https://github.com/pytorch/tensordict/).

*   **Example Code:**
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
   
    <img src="https://github.com/pytorch/rl/blob/main/docs/source/_static/img/rollout.gif" alt="Rollout Example">

## Features (Detailed)

*   **Environment Interface:** A unified interface for environments with support for popular libraries (OpenAI Gym, DeepMind Control Lab) and state-less execution. Batched environments for parallel execution and PyTorch-first tensor-specification classes are also provided. See  [environment API documentation](https://pytorch.org/rl/stable/reference/envs.html) and [tutorial](https://pytorch.org/rl/stable/tutorials/pendulum.html).
*   **Data Collectors:** Multiprocess and distributed data collectors for synchronous and asynchronous data collection.  See [distributed collector examples](https://github.com/pytorch/rl/blob/main/examples/distributed/collectors).
*   **Replay Buffers:** Efficient and generic replay buffers with modularized storage, including wrappers for offline RL datasets.
*   **Environment Transforms:** Cross-library environment transforms executed on device and in a vectorized fashion.
*   **Models and Architectures:** Various pre-built architectures and models (actor-critic).
*   **Exploration Wrappers:** Wrappers and modules for easy swapping between exploration and exploitation.
*   **Loss Modules:** Efficient loss modules and highly vectorized functional return and advantage computation.
*   **Trainer Class:** A generic trainer class that executes the training loop with a hooking mechanism for logging and data transformation.
*   **Recipes:** Recipes to build models corresponding to the deployed environment.

## Examples, Tutorials, and Demos

Explore these resources for practical implementation:

*   **State-of-the-Art Implementations:** [DQN, DDPG, IQL, CQL, TD3, TD3+BC, A2C, PPO, SAC, REDQ, Dreamer v1, Decision Transformers, CrossQ, Gail, Impala, Multi-Agent IQL, DDPG, PPO, QMIX-VDN, SAC, RLHF, LLM API (GRPO) and many more!](https://github.com/pytorch/rl/blob/main/sota-implementations/)
*   **Code Examples:** [LLM API & GRPO, RLHF, Memory-mapped replay buffers.](examples/)
*   **Tutorials and Demos:**  [Explore them here](https://pytorch.org/rl/stable#tutorials)

## Citation

If you use TorchRL, cite this work:
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

### Set up Environment:

```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies:

Choose your PyTorch Installation (Nightly, Stable). TorchRL supports various installation methods including pip and local builds:

*   **Install the latest release**:
    ```bash
    pip3 install torchrl
    ```
*   **Install Nightly Build:**
    ```bash
    pip3 install tensordict-nightly torchrl-nightly
    ```
    (Requires PyTorch nightly)
*   **Install Specific PyTorch Version (Example):**
    ```bash
    pip3 install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip3 install torchrl
    ```
*   **For a detailed list of commands,** see [here](https://pytorch.org/get-started/locally/)
### Install Optional Dependencies:
```bash
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher
pip3 install "moviepy<2.0.0"
pip3 install dm_control
pip3 install "gym[atari]" "gym[accept-rom-license]" pygame
pip3 install pytest pyyaml pytest-instafail
pip3 install tensorboard
pip3 install wandb
```

## Asking a Question

*   **Report Bugs:**  Raise an issue in the [TorchRL repository](https://github.com/pytorch/rl).
*   **General RL Questions:**  Post on the  [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome! Follow the guide [here](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) to get started.  See [open contributions](https://github.com/pytorch/rl/issues/509) for areas of focus.  Install [pre-commit hooks](https://pre-commit.com/) to check linting before commits.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.