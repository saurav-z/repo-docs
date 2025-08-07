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

[**TorchRL**](https://github.com/pytorch/rl) is an open-source library that empowers researchers and developers to build state-of-the-art Reinforcement Learning (RL) models efficiently using PyTorch.

## Key Features

*   **Python-First Design**: Enjoy an intuitive and flexible Python-first interface.
*   **High Performance**: Built for demanding RL research, TorchRL is optimized for speed.
*   **Modular Architecture**: Easily customize and extend components to fit your unique needs.
*   **Comprehensive Documentation**: Clear, thorough documentation to get you up and running quickly.
*   **Rigorous Testing**: Benefit from a reliably tested library, ensuring stability and accuracy.
*   **Reusable Functionals**: Leverage a set of reusable functions for cost functions, returns, and data processing.
*   **LLM API:** A comprehensive framework for language model fine-tuning, supporting RLHF, supervised fine-tuning, and tool-augmented training.

## What's New: LLM API for Language Model Fine-tuning

TorchRL now includes a groundbreaking **LLM API** for post-training and fine-tuning of language models! This framework streamlines RLHF, supervised fine-tuning, and tool-augmented training with:

*   **Unified LLM Wrappers:** Seamless integration with Hugging Face models and vLLM inference engines.
*   **Advanced Conversation Management:** Utilize the `History` class for multi-turn dialogue with automatic chat template detection.
*   **Tool Integration:** Built-in support for Python code execution, function calling, and custom tool transforms.
*   **Specialized Objectives:** Access GRPO (Group Relative Policy Optimization) and SFT (Supervised Fine-tuning) loss functions optimized for language models.
*   **High-Performance Collectors:** Leverage async data collection with distributed training support.
*   **Flexible Environments:** Utilize a transform-based architecture for reward computation, data loading, and conversation augmentation.

For more details, check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo).

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

Accelerate your RL journey with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

*   Explore the comprehensive [TorchRL documentation](https://pytorch.org/rl) for tutorials and API reference.
*   Access the [RL knowledge base](https://pytorch.org/rl/stable/reference/knowledge_base.html) to debug code and learn RL basics.
*   Watch introductory videos to learn more about the library:
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL has been successfully used in various fields. Some of the spotlight publications are:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Simplified RL Codebases with `TensorDict`

TorchRL leverages `TensorDict` to streamline RL codebases.
With this tool, you can write a *complete PPO training script in less than 100 lines of code*!

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

  TensorDict supports multiple tensor operations on its device and shape
  (the shape of TensorDict, or its batch size, is the common arbitrary N first dimensions of all its contained tensors):

  <details>
    <summary>Code</summary>

  ```python
  # stack and cat
  tensordict = torch.stack(list_of_tensordicts, 0)
  tensordict = torch.cat(list_of_tensordicts, 0)
  # reshape
  tensordict = tensordict.view(-1)
  tensordict = tensordict.permute(0, 2, 1)
  tensordict = tensordict.unsqueeze(-1)
  tensordict = tensordict.squeeze(-1)
  # indexing
  tensordict = tensordict[:2]
  tensordict[:, 2] = sub_tensordict
  # device and memory location
  tensordict.cuda()
  tensordict.to("cuda:1")
  tensordict.share_memory_()
  ```
  </details>

Check [TensorDict tutorials](https://pytorch.github.io/tensordict/) to
  learn more!

## Features

*   **Environment Interface:** Standardized interface for environments, supporting common libraries (OpenAI Gym, DeepMind Control Lab, etc.). Batched environments enable parallel execution.
*   **Data Collectors**: Multiprocess and distributed data collectors for synchronous and asynchronous data collection.
*   **Replay Buffers**: Efficient and generic replay buffers with modularized storage, including wrappers for offline RL datasets.
*   **Environment Transforms**: Cross-library transforms for data processing, executed on device in a vectorized fashion.
*   **Models and Architectures**: A wide range of architectures and models (e.g. actor-critic).
*   **Exploration Wrappers**: Easily swap between exploration and exploitation strategies.
*   **Loss Modules**: Efficient loss modules and vectorized return/advantage computation.
*   **Trainer Class**: A generic trainer class that executes the training loop and supports logging/data transformation.
*   **Recipes**: Provides a series of recipes to build models that correspond to the environment being deployed.

## Examples, Tutorials, and Demos

*   Explore a suite of [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) for various algorithms.
*   Access [code examples](examples/) for toy code snippets and training scripts.
*   Follow [tutorials and demos](https://pytorch.org/rl/stable#tutorials) to discover the library's capabilities.

## Citation

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

Follow these steps to get started:

### Create a Virtual Environment
```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```
OR
```
conda create --name torchrl python=3.9
conda activate torchrl
```

### Install Dependencies

Follow instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to install PyTorch.

Then, install TorchRL:
```bash
pip3 install torchrl
```

For specific use cases and nightly builds, see detailed instructions in the original README.

### Optional Dependencies

Install optional libraries based on your needs:
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

If you have any questions about the library, submit an issue in this repo. For more generic questions regarding RL in PyTorch, post it on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

We welcome contributions! See the detailed contribution guide [here](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).
A list of open contributions can be found in [here](https://github.com/pytorch/rl/issues/509).

## Disclaimer

TorchRL is a beta feature and breaking changes may occur, but will be introduced with a deprecation
warranty after a few release cycles.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.