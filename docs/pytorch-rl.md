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

[**TorchRL**](https://github.com/pytorch/rl) is an open-source library built on PyTorch, empowering researchers and developers to build and deploy reinforcement learning (RL) solutions.

## Key Features

*   **Python-First Design:** Enjoy an intuitive and flexible experience with Python as the primary language.
*   **Efficiency:** Benefit from optimized performance, perfect for demanding RL research applications.
*   **Modularity and Extensibility:** Customize and extend your RL pipelines with a highly modular architecture.
*   **Comprehensive Documentation:** Easily understand and utilize the library with thorough documentation and tutorials.
*   **Reliability:** Rely on a rigorously tested codebase for stability and performance.
*   **Reusable Components:** Leverage a suite of reusable functions for costs, returns, and data processing.
*   **LLM API:** Build advanced language models with a complete framework for fine-tuning, RLHF, and tool integration.

## Core Functionality

*   **Environments:** Offers a unified interface for environments, supporting popular libraries (OpenAI Gym, DeepMind Control, etc.) and state-less execution. The batched environment containers allow parallel execution. Tensor specifications provide a common class.
*   **Data Collection:** Includes multiprocess and distributed data collectors that work synchronously or asynchronously, facilitating ultra-fast data collection.
*   **Replay Buffers:** Offers efficient and generic replay buffers with modular storage, including wrappers around common datasets for offline RL.
*   **Environment Transforms:** Provides cross-library environment transforms, executed on-device and in a vectorized fashion, to prepare data for the agent.
*   **Models and Architectures:** Offers various models (actor-critic, etc.).
*   **Exploration Tools:** Makes it easy to swap between exploration and exploitation.
*   **Loss Modules and Functional Computation:** Features efficient loss modules and highly vectorized functional return and advantage computation.
*   **Trainer Class:** Executes training loops and supports logging and data transformation.
*   **Recipes:** Provides ready-to-use models.

### LLM API Deep Dive

TorchRL's LLM API is a complete framework for LLM fine-tuning.

**Key features**:

*   **Unified LLM Wrappers:** Seamless integration with Hugging Face models and vLLM inference engines - more to come!
*   **Conversation Management:** Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
*   **Tool Integration:** [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
*   **Specialized Objectives:** [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
*   **High-Performance Collectors:** [Async data collection](torchrl/collectors/llm/) with distributed training support
*   **Flexible Environments:** Transform-based architecture for reward computation, data loading, and conversation augmentation

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

## Get Started

Refer to our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to ramp up quickly with the basic features of the library!

## Documentation and Knowledge Base

Explore the comprehensive [TorchRL documentation](https://pytorch.org/rl/) for tutorials, API reference, and a RL knowledge base to aid in debugging and understanding RL fundamentals.

- [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
- [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
- [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL has been applied in a wide array of domains. Here are a few examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): RL of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent RL
*   [BricksRL](https://arxiv.org/abs/2406.17490): Democratizing Robotics and RL Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): RL in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): RL for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): Unified framework for robot learning

## Simplify Your RL Code with TensorDict

TorchRL leverages [`TensorDict`](https://github.com/pytorch/tensordict/) to simplify RL codebases.  This data structure streamlines your code and allows you to write complete PPO training scripts in under 100 lines.

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

## Features

*   A common [interface for environments](https://github.com/pytorch/rl/blob/main/torchrl/envs)
  which supports common libraries (OpenAI gym, deepmind control lab, etc.)<sup>(1)</sup> and state-less execution
  (e.g. Model-based environments).
  The [batched environments](https://github.com/pytorch/rl/blob/main/torchrl/envs/batched_envs.py) containers allow parallel execution<sup>(2)</sup>.
  A common PyTorch-first class of [tensor-specification class](https://github.com/pytorch/rl/blob/main/torchrl/data/tensor_specs.py) is also provided.
  TorchRL's environments API is simple but stringent and specific. Check the
  [documentation](https://pytorch.org/rl/stable/reference/envs.html)
  and [tutorial](https://pytorch.org/rl/stable/tutorials/pendulum.html) to learn more!
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

*   Multiprocess and distributed [data collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py)<sup>(2)</sup>
  that work synchronously or asynchronously.
  Through the use of TensorDict, TorchRL's training loops are made very similar
  to regular training loops in supervised
  learning (although the "dataloader" -- read data collector -- is modified on-the-fly):
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

  Check our [distributed collector examples](https://github.com/pytorch/rl/blob/main/examples/distributed/collectors) to
  learn more about ultra-fast data collection with TorchRL.

*   Efficient<sup>(2)</sup> and generic<sup>(1)</sup> [replay buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py) with modularized storage:
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

  Replay buffers are also offered as wrappers around common datasets for *offline RL*:
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

*   Cross-library [environment transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py)<sup>(1)</sup>,
  executed on device and in a vectorized fashion<sup>(2)</sup>,
  which process and prepare the data coming out of the environments to be used by the agent:
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
  Other transforms include: reward scaling (`RewardScaling`), shape operations (concatenation of tensors, unsqueezing etc.), concatenation of
  successive operations (`CatFrames`), resizing (`Resize`) and many more.

  Unlike other libraries, the transforms are stacked as a list (and not wrapped in each other), which makes it
  easy to add and remove them at will:
  ```python
  env.insert_transform(0, NoopResetEnv())  # inserts the NoopResetEnv transform at the index 0
  ```
  Nevertheless, transforms can access and execute operations on the parent environment:
  ```python
  transform = env.transform[1]  # gathers the second transform of the list
  parent_env = transform.parent  # returns the base environment of the second transform, i.e. the base env + the first transform
  ```
  </details>

*   Various tools for distributed learning (e.g. [memory mapped tensors](https://github.com/pytorch/tensordict/blob/main/tensordict/memmap.py))<sup>(2)</sup>;
*   Various [architectures](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/) and models (e.g. [actor-critic](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/actors.py))<sup>(1)</sup>:
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

*   Exploration [wrappers](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py) and
  [modules](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/exploration.py) to easily swap between exploration and exploitation<sup>(1)</sup>:
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

*   A series of efficient [loss modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives)
  and highly vectorized
  [functional return and advantage](https://github.com/pytorch/rl/blob/main/torchrl/objectives/value/functional.py)
  computation.

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

*   A generic [trainer class](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py)<sup>(1)</sup> that
  executes the aforementioned training loop. Through a hooking mechanism,
  it also supports any logging or data transformation operation at any given
  time.

*   Various [recipes](https://github.com/pytorch/rl/blob/main/torchrl/trainers/helpers/models.py) to build models that
    correspond to the environment being deployed.

*   **LLM API**: Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends,
  conversation management with automatic chat template detection, tool integration (Python execution, function calling),
  specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning,
  and tool-augmented training scenarios.
  <details>
    <summary>Code</summary>

  ```python
  from torchrl.envs.llm import ChatEnv
  from torchrl.modules.llm import TransformersWrapper
  from torchrl.envs.llm.transforms import PythonInterpreter
  
  # Create environment with tool execution
  env = ChatEnv(
      tokenizer=tokenizer,
      system_prompt="You can execute Python code.",
      batch_size=[1]
  ).append_transform(PythonInterpreter())
  
  # Wrap language model for training
  llm = TransformersWrapper(
      model=model,
      tokenizer=tokenizer,
      input_mode="history"
  )
  
  # Multi-turn conversation with tool use
  obs = env.reset(TensorDict({"query": "Calculate 2+2"}, batch_size=[1]))
  llm_output = llm(obs)  # Generates response
  obs = env.step(llm_output)  # Environment processes response
  ```
  </details>

## Examples, Tutorials, and Demos

Explore the [examples directory](https://github.com/pytorch/rl/blob/main/sota-implementations/) for state-of-the-art implementations, including:

*   **DQN, DDPG, IQL, CQL, TD3, A2C, PPO, SAC, REDQ, Dreamer v1, Decision Transformers, and more.**

[Code examples](examples/) displaying toy code snippets and training scripts are also available
- [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
- [RLHF](examples/rlhf)
- [Memory-mapped replay buffers](examples/torchrl_features)

Check the [examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) directory for more details
about handling the various configuration settings.

We also provide [tutorials and demos](https://pytorch.org/rl/stable#tutorials) that give a sense of
what the library can do.

## Citation

If you use TorchRL in your research, please cite the following:

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

Follow these steps to get started with TorchRL:

### 1. Create a Virtual Environment:

```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

or a conda environment:
```
conda create --name torchrl python=3.9
conda activate torchrl
```

### 2. Install PyTorch:

Install the appropriate PyTorch version (stable or nightly) for your needs.  Refer to the [PyTorch Get Started page](https://pytorch.org/get-started/locally/) for detailed instructions, including commands like `pip3`.

### 3. Install TorchRL:

```bash
pip3 install torchrl
```

For the nightly build, install using:
```bash
pip3 install tensordict-nightly torchrl-nightly
```

**Important:** Ensure compatibility between your PyTorch and TorchRL versions.  Refer to the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md) for potential issues and workarounds.

### 4. Install Optional Dependencies:

Install these libraries, depending on your needs:
```bash
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher
pip3 install "moviepy<2.0.0"
pip3 install dm_control
pip3 install "gym[atari]" "gym[accept-rom-license]" pygame
pip3 install pytest pyyaml pytest-instafail
pip3 install tensorboard
pip3 install wandb
```

## Get Involved

*   Report bugs and ask questions by raising an issue in this repo.
*   Discuss RL in PyTorch on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).
*   Contribute by forking the repository, submitting issues and PRs, and consulting the detailed [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).  See [open contributions](https://github.com/pytorch/rl/issues/509) to assist.

## License and Disclaimer

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.  It's a PyTorch beta feature, and breaking changes are possible, with deprecation notices expected.