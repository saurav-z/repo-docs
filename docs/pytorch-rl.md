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

# TorchRL: A PyTorch Library for Modern Reinforcement Learning

**TorchRL** is a cutting-edge, open-source library built on PyTorch, empowering researchers and developers to build and experiment with Reinforcement Learning (RL) algorithms. Explore the original repo [here](https://github.com/pytorch/rl)!

## Key Features

*   **Python-First Design:** Prioritizes Python for ease of use and flexibility in RL projects.
*   **Optimized for Performance:** Provides efficient implementations to support demanding RL research applications.
*   **Modular and Extensible:** Offers a highly modular architecture for easy swapping, transforming, and creating new components.
*   **Comprehensive Documentation:** Includes thorough documentation, ensuring users can quickly understand and utilize the library.
*   **Rigorous Testing:** Ensures reliability and stability through extensive testing.
*   **Reusable Functionals:** Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   **LLM API**: Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, 
  conversation management with automatic chat template detection, tool integration (Python execution, function calling), 
  specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning, 
  and tool-augmented training scenarios.

## What's New: LLM API - Complete Framework for Language Model Fine-tuning

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

*   🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
*   💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
*   🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
*   🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
*   ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
*   🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

The LLM API follows TorchRL's modular design principles, allowing you to mix and match components for your specific use case. Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

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

Dive into RL with TorchRL by checking out the [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly ramp up with the basic features of the library!

## Documentation and Knowledge Base

*   **Documentation:** Find comprehensive documentation, tutorials, and the API reference [here](https://pytorch.org/rl).
*   **Knowledge Base:** Access a RL knowledge base to help you debug your code and learn the basics of RL [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).
*   **Introductory Videos:**
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is versatile and applicable across various fields. Here are some example publications:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Simplifying RL Codebases with TensorDict

TorchRL leverages [`TensorDict`](https://github.com/pytorch/tensordict/), a powerful data structure, to streamline RL code. With TensorDict, you can write a complete PPO training script in a concise and manageable manner.

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

## Key Features in Detail

*   **Environment Abstraction:** Provides a unified interface for environments. Supports common libraries (OpenAI Gym, DeepMind Control Lab, etc.) and state-less execution. Batched environments allow parallel execution. A common PyTorch-first class of tensor-specification class is also provided.
    <details>
    <summary>Code</summary>

  ```python
  env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
  env_parallel = ParallelEnv(4, env_make)  # creates 4 envs in parallel
  tensordict = env_parallel.rollout(max_steps=20, policy=None)  # random rollout (no policy given)
  assert tensordict.shape == [4, 20]  # 4 envs, 20 steps rollout
  env_parallel.action_spec.is_in(tensordict["action"])  # spec check returns True
  ```
  </details>

*   **Data Collection:** Includes multiprocess and distributed data collectors that work synchronously or asynchronously. TorchRL's training loops are very similar to regular training loops in supervised learning.
    <details>
    <summary>Code</summary>

  ```python
  env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
  collector = MultiaSyncDataCollector(
      [env_make, env_make],
      policy=policy,
      devices=["cuda:0", "cuda:0"],
      total_frames=10000,
      frames_per_batch=50,
      ...
  )
  for i, tensordict_data in enumerate(collector):
      loss = loss_module(tensordict_data)
      loss.backward()
      optim.step()
      optim.zero_grad()
      collector.update_policy_weights_()
  ```
  </details>

*   **Replay Buffers:** Offers efficient and generic replay buffers with modularized storage. Replay buffers are also offered as wrappers around common datasets for *offline RL*.
    <details>
    <summary>Code</summary>

  ```python
  storage = LazyMemmapStorage(  # memory-mapped (physical) storage
      cfg.buffer_size,
      scratch_dir="/tmp/"
  )
  buffer = TensorDictPrioritizedReplayBuffer(
      alpha=0.7,
      beta=0.5,
      collate_fn=lambda x: x,
      pin_memory=device != torch.device("cpu"),
      prefetch=10,  # multi-threaded sampling
      storage=storage
  )
  ```
  </details>
  <details>
    <summary>Code</summary>

  ```python
  from torchrl.data.replay_buffers import SamplerWithoutReplacement
  from torchrl.data.datasets.d4rl import D4RLExperienceReplay
  data = D4RLExperienceReplay(
      "maze2d-open-v0",
      split_trajs=True,
      batch_size=128,
      sampler=SamplerWithoutReplacement(drop_last=True),
  )
  for sample in data:  # or alternatively sample = data.sample()
      fun(sample)
  ```
  </details>

*   **Environment Transforms:** Offers cross-library environment transforms, executed on device and in a vectorized fashion, which process and prepare the data coming out of the environments.
    <details>
    <summary>Code</summary>

  ```python
  env_make = lambda: GymEnv("Pendulum-v1", from_pixels=True)
  env_base = ParallelEnv(4, env_make, device="cuda:0")  # creates 4 envs in parallel
  env = TransformedEnv(
      env_base,
      Compose(
          ToTensorImage(),
          ObservationNorm(loc=0.5, scale=1.0)),  # executes the transforms once and on device
  )
  tensordict = env.reset()
  assert tensordict.device == torch.device("cuda:0")
  ```
  </details>

*   **Model Architectures:** Provides various architectures and models (e.g. actor-critic).
    <details>
    <summary>Code</summary>

  ```python
  # create an nn.Module
  common_module = ConvNet(
      bias_last_layer=True,
      depth=None,
      num_cells=[32, 64, 64],
      kernel_sizes=[8, 4, 3],
      strides=[4, 2, 1],
  )
  # Wrap it in a SafeModule, indicating what key to read in and where to
  # write out the output
  common_module = SafeModule(
      common_module,
      in_keys=["pixels"],
      out_keys=["hidden"],
  )
  # Wrap the policy module in NormalParamsWrapper, such that the output
  # tensor is split in loc and scale, and scale is mapped onto a positive space
  policy_module = SafeModule(
      NormalParamsWrapper(
          MLP(num_cells=[64, 64], out_features=32, activation=nn.ELU)
      ),
      in_keys=["hidden"],
      out_keys=["loc", "scale"],
  )
  # Use a SafeProbabilisticTensorDictSequential to combine the SafeModule with a
  # SafeProbabilisticModule, indicating how to build the
  # torch.distribution.Distribution object and what to do with it
  policy_module = SafeProbabilisticTensorDictSequential(  # stochastic policy
      policy_module,
      SafeProbabilisticModule(
          in_keys=["loc", "scale"],
          out_keys="action",
          distribution_class=TanhNormal,
      ),
  )
  value_module = MLP(
      num_cells=[64, 64],
      out_features=1,
      activation=nn.ELU,
  )
  # Wrap the policy and value funciton in a common module
  actor_value = ActorValueOperator(common_module, policy_module, value_module)
  # standalone policy from this
  standalone_policy = actor_value.get_policy_operator()
  ```
  </details>

*   **Exploration:** Includes exploration wrappers and modules to easily swap between exploration and exploitation.
    <details>
    <summary>Code</summary>

  ```python
  policy_explore = EGreedyWrapper(policy)
  with set_exploration_type(ExplorationType.RANDOM):
      tensordict = policy_explore(tensordict)  # will use eps-greedy
  with set_exploration_type(ExplorationType.DETERMINISTIC):
      tensordict = policy_explore(tensordict)  # will not use eps-greedy
  ```
  </details>

*   **Loss Modules:** Provides a series of efficient loss modules and highly vectorized functional return and advantage computation.
    <details>
    <summary>Code</summary>

  ### Loss modules
  ```python
  from torchrl.objectives import DQNLoss
  loss_module = DQNLoss(value_network=value_network, gamma=0.99)
  tensordict = replay_buffer.sample(batch_size)
  loss = loss_module(tensordict)
  ```

  ### Advantage computation
  ```python
  from torchrl.objectives.value.functional import vec_td_lambda_return_estimate
  advantage = vec_td_lambda_return_estimate(gamma, lmbda, next_state_value, reward, done, terminated)
  ```

  </details>

*   **Trainer Class:** Offers a generic trainer class that executes the training loop and supports logging and data transformation.
*   **Recipes:** Provides various recipes to build models that correspond to the environment being deployed.

## Examples, Tutorials, and Demos

Explore the capabilities of TorchRL with these resources:

*   **State-of-the-Art Implementations:**
    *   [DQN](https://github.com/pytorch/rl/blob/main/sota-implementations/dqn)
    *   [DDPG](https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py)
    *   [IQL](https://github.com/pytorch/rl/blob/main/sota-implementations/iql/)
    *   [CQL](https://github.com/pytorch/rl/blob/main/sota-implementations/cql/cql_offline.py)
    *   [TD3](https://github.com/pytorch/rl/blob/main/sota-implementations/td3/td3.py)
    *   [TD3+BC](https://github.com/pytorch/rl/blob/main/sota-implementations/td3_bc/td3_bc.py)
    *   [A2C](https://github.com/pytorch/rl/blob/main/examples/a2c/)
    *   [PPO](https://github.com/pytorch/rl/blob/main/sota-implementations/ppo/)
    *   [SAC](https://github.com/pytorch/rl/blob/main/sota-implementations/sac/sac.py)
    *   [REDQ](https://github.com/pytorch/rl/blob/main/sota-implementations/redq/redq.py)
    *   [Dreamer v1](https://github.com/pytorch/rl/blob/main/sota-implementations/dreamer/dreamer.py)
    *   [Decision Transformers](https://github.com/pytorch/rl/blob/main/sota-implementations/decision_transformer)
    *   [CrossQ](https://github.com/pytorch/rl/blob/main/sota-implementations/crossq)
    *   [Gail](https://github.com/pytorch/rl/blob/main/sota-implementations/gail)
    *   [Impala](https://github.com/pytorch/rl/blob/main/sota-implementations/impala)
    *   [IQL (MARL)](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py)
    *   [DDPG (MARL)](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/maddpg_iddpg.py)
    *   [PPO (MARL)](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/mappo_ippo.py)
    *   [QMIX-VDN (MARL)](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/qmix_vdn.py)
    *   [SAC (MARL)](https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/sac.py)
    *   [RLHF](examples/rlhf)
    *   [LLM API (GRPO)](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   **Code Examples:**
    *   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
    *   [RLHF](examples/rlhf)
    *   [Memory-mapped replay buffers](examples/torchrl_features)
*   **Tutorials and Demos:** [Tutorials and demos](https://pytorch.org/rl/stable#tutorials)

## Citation

If you use TorchRL in your research, please cite it using the following BibTeX entry:

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

### Prerequisites:

*   Python 3.9 or higher
*   PyTorch (installation instructions below)

### Installation Steps:

1.  **Create a virtual environment:**
    ```bash
    python -m venv torchrl
    source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
    ```
    Or create a conda environment:
    ```bash
    conda create --name torchrl python=3.9
    conda activate torchrl
    ```

2.  **Install PyTorch:** Choose the appropriate installation command based on your system (see [PyTorch installation instructions](https://pytorch.org/get-started/locally/)).

3.  **Install TorchRL:**
    ```bash
    pip3 install torchrl
    ```
    For more specific instructions, including nightly builds and local builds, see the original README [here](https://github.com/pytorch/rl).

4.  **Install Optional Dependencies:** (Recommended for full functionality)
    ```bash
    pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher
    pip3 install "moviepy<2.0.0" # rendering
    pip3 install dm_control # deepmind control suite
    pip3 install "gym[atari]" "gym[accept-rom-license]" pygame # gym, atari games
    pip3 install pytest pyyaml pytest-instafail # tests
    pip3 install tensorboard
    pip3 install wandb # wandb
    ```

## Contributing

We welcome contributions! Please review the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) and the list of open contributions [here](https://github.com/pytorch/rl/issues/509).

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.