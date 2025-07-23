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

# TorchRL: A PyTorch Library for Data-Driven Decision-Making

TorchRL is an open-source library built on PyTorch that offers a comprehensive toolkit for Reinforcement Learning (RL) research and development.

[**Documentation**](https://pytorch.org/rl/) | [**TensorDict**](#writing-simplified-and-portable-rl-codebase-with-tensordict) |
[**Features**](#key-features) | [**Examples, tutorials and demos**](#examples-tutorials-and-demos) | [**Citation**](#citation) | [**Installation**](#installation) |
[**Asking a question**](#asking-a-question) | [**Contributing**](#contributing)

## 🚀 What's New

### LLM API - Complete Framework for Language Model Fine-tuning

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

- 🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
- 💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
- 🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
- 🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
- ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
- 🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

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

## Key Features

*   **Python-first:** Designed with Python as the primary language for ease of use and flexibility.
*   **Efficient:** Optimized for performance to support demanding RL research applications.
*   **Modular, Customizable, Extensible:** Highly modular architecture allows for easy swapping, transformation, or creation of new components.
*   **Well-Documented:** Comprehensive documentation ensures that users can quickly understand and utilize the library.
*   **Rigorously Tested:** Extensive testing to ensure reliability and stability.
*   **Reusable Functionals:** Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   **LLM API:** Complete framework for Language Model fine-tuning with unified wrappers, conversation management, tool integration, specialized objectives, and high-performance collectors.

### Design Principles

*   **Alignment with PyTorch Ecosystem:** Follows the structure and conventions of popular PyTorch libraries.
*   **Minimal Dependencies:** Requires only Python standard library, NumPy, and PyTorch; optional dependencies for common environment libraries and datasets.

Read the [full paper](https://arxiv.org/abs/2306.00577) for a more curated description of the library.

## Getting Started

Begin exploring TorchRL with the [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly ramp up with the basic features of the library!

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

*   Access the comprehensive TorchRL documentation [here](https://pytorch.org/rl), including tutorials and API reference.
*   Explore the RL knowledge base to debug code and learn the fundamentals of RL [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).
*   Watch introductory videos to learn more about the library:
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is a versatile library used in diverse fields. Here are some examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL uses [`TensorDict`](https://github.com/pytorch/tensordict/) to streamline your RL codebase and make it easier to recycle code across different settings.  With TensorDict, complete PPO training scripts can be written in a fraction of the code.

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

The [environment API](https://pytorch.org/rl/stable/reference/envs.html) relies on TensorDict to transfer data during rollouts.
![Alt Text](https://github.com/pytorch/rl/blob/main/docs/source/_static/img/rollout.gif)

TensorDict simplifies code reuse across various RL settings.

  <details>
    <summary>Code</summary>
  
  Here's how to code a rollout in TorchRL:

  ```diff
  - obs, done = env.reset()
  + tensordict = env.reset()
  policy = SafeModule(
      model,
      in_keys=["observation_pixels", "observation_vector"],
      out_keys=["action"],
  )
  out = []
  for i in range(n_steps):
  -     action, log_prob = policy(obs)
  -     next_obs, reward, done, info = env.step(action)
  -     out.append((obs, next_obs, action, log_prob, reward, done))
  -     obs = next_obs
  +     tensordict = policy(tensordict)
  +     tensordict = env.step(tensordict)
  +     out.append(tensordict)
  +     tensordict = step_mdp(tensordict)  # renames next_observation_* keys to observation_*
  - obs, next_obs, action, log_prob, reward, done = [torch.stack(vals, 0) for vals in zip(*out)]
  + out = torch.stack(out, 0)  # TensorDict supports multiple tensor operations
  ```
  </details>

  Using TensorDict, TorchRL abstracts the input/output signatures of modules, environments, and losses,
  making it easier to reuse across settings.

  <details>
    <summary>Code</summary>

  Here's an off-policy training loop in TorchRL:
  
  ```diff
  - for i, (obs, next_obs, action, hidden_state, reward, done) in enumerate(collector):
  + for i, tensordict in enumerate(collector):
  -     replay_buffer.add((obs, next_obs, action, log_prob, reward, done))
  +     replay_buffer.add(tensordict)
      for j in range(num_optim_steps):
  -         obs, next_obs, action, hidden_state, reward, done = replay_buffer.sample(batch_size)
  -         loss = loss_fn(obs, next_obs, action, hidden_state, reward, done)
  +         tensordict = replay_buffer.sample(batch_size)
  +         loss = loss_fn(tensordict)
          loss.backward()
          optim.step()
          optim.zero_grad()
  ```
  This training loop is reusable across algorithms as it makes minimal assumptions about data structure.
  </details>

  TensorDict supports tensor operations on its device and shape:

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

TensorDict includes a dedicated [`tensordict.nn`](https://pytorch.github.io/tensordict/reference/nn.html) module for model creation.  It's also `functorch` and `torch.compile` compatible!

  <details>
    <summary>Code</summary>

  ```diff
  transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
  + td_module = SafeModule(transformer_model, in_keys=["src", "tgt"], out_keys=["out"])
  src = torch.rand((10, 32, 512))
  tgt = torch.rand((20, 32, 512))
  + tensordict = TensorDict({"src": src, "tgt": tgt}, batch_size=[20, 32])
  - out = transformer_model(src, tgt)
  + td_module(tensordict)
  + out = tensordict["out"]
  ```

  `TensorDictSequential` allows to branch sequences of `nn.Module` instances in a highly modular way.
  For instance, here is an implementation of a transformer using the encoder and decoder blocks:
  ```python
  encoder_module = TransformerEncoder(...)
  encoder = TensorDictSequential(encoder_module, in_keys=["src", "src_mask"], out_keys=["memory"])
  decoder_module = TransformerDecoder(...)
  decoder = TensorDictModule(decoder_module, in_keys=["tgt", "memory"], out_keys=["output"])
  transformer = TensorDictSequential(encoder, decoder)
  assert transformer.in_keys == ["src", "src_mask", "tgt"]
  assert transformer.out_keys == ["memory", "output"]
  ```

  `TensorDictSequential` allows to isolate subgraphs by querying a set of desired input / output keys:
  ```python
  transformer.select_subsequence(out_keys=["memory"])  # returns the encoder
  transformer.select_subsequence(in_keys=["tgt", "memory"])  # returns the decoder
  ```
  </details>

  Check [TensorDict tutorials](https://pytorch.github.io/tensordict/) to learn more!

## Features

*   **Environments:** A common [interface for environments](https://github.com/pytorch/rl/blob/main/torchrl/envs) supporting common libraries (OpenAI gym, deepmind control lab, etc.) and state-less execution, with batched environments for parallel execution and a PyTorch-first tensor-specification class. TorchRL's environments API is simple but stringent. Check the [documentation](https://pytorch.org/rl/stable/reference/envs.html) and [tutorial](https://pytorch.org/rl/stable/tutorials/pendulum.html) to learn more!
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
*   **Data Collectors:** Multiprocess and distributed [data collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py) that work synchronously or asynchronously. TorchRL's training loops are made very similar to regular training loops in supervised learning (although the "dataloader" -- read data collector -- is modified on-the-fly):
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
  Check our [distributed collector examples](https://github.com/pytorch/rl/blob/main/examples/distributed/collectors) to learn more about ultra-fast data collection with TorchRL.
*   **Replay Buffers:** Efficient and generic [replay buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py) with modularized storage.
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
*   **Environment Transforms:** Cross-library [environment transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py), executed on device and in a vectorized fashion, processing data from environments for the agent.
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
*   **Tools for Distributed Learning:** Including [memory mapped tensors](https://github.com/pytorch/tensordict/blob/main/tensordict/memmap.py).
*   **Architectures and Models:** Various [architectures](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/) and models (e.g. [actor-critic](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/actors.py)).
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
*   **Exploration Wrappers and Modules:** Exploration [wrappers](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py) and [modules](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/exploration.py) to easily swap between exploration and exploitation.
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
*   **Loss Modules:** A series of efficient [loss modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives) and highly vectorized [functional return and advantage](https://github.com/pytorch/rl/blob/main/torchrl/objectives/value/functional.py) computation.
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
*   **Trainer Class:** A generic [trainer class](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py) that executes the training loop, with a hooking mechanism for logging or data transformation.
*   **Recipes:** Various [recipes](https://github.com/pytorch/rl/blob/main/torchrl/trainers/helpers/models.py) to build models that correspond to the environment being deployed.
*   **LLM API:** Complete framework for Language Model fine-tuning with unified wrappers, conversation management, tool integration, specialized objectives, and high-performance collectors.

If you'd like to add features, see the [contribution page](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).

## Examples, Tutorials and Demos

Explore the [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) with these features:

<table>
  <tr>
   <td><strong>Algorithm</strong>
   </td>
   <td><strong>Compile Support**</strong>
   </td>
   <td><strong>Tensordict-free API</strong>
   </td>
   <td><strong>Modular Losses</strong>
   </td>
   <td><strong>Continuous and Discrete</strong>
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/dqn">DQN</a>
   </td>
   <td> 1.9x
   </td>
   <td> +
   </td>
   <td> NA
   </td>
   <td> + (through <a href="https://pytorch.org/rl/stable/reference/generated/torchrl.envs.transforms.ActionDiscretizer.html?highlight=actiondiscretizer">ActionDiscretizer</a> transform)
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py">DDPG</a>
   </td>
   <td> 1.87x
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/iql/">IQL</a>
   </td>
   <td> 3.22x
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/cql/cql_offline.py">CQL</a>
   </td>
   <td> 2.68x
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/td3/td3.py">TD3</a>
   </td>
   <td> 2.27x
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td>
    <a href="https://github.com/pytorch/rl/blob/main/sota-implementations/td3_bc/td3_bc.py">TD3+BC</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td>
    <a href="https://github.com/pytorch/rl/blob/main/examples/a2c/">A2C</a>
   </td>
   <td> 2.67x
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td>
    <a href="https://github.com/pytorch/rl/blob/main/sota-implementations/ppo/">PPO</a>
   </td>
   <td> 2.42x
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/sac/sac.py">SAC</a>
   </td>
   <td> 2.62x
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/redq/redq.py">REDQ</a>
   </td>
   <td> 2.28x
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/dreamer/dreamer.py">Dreamer v1</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> + (<a href="https://pytorch.org/rl/stable/reference/objectives.html#dreamer">different classes</a>)
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/decision_transformer">Decision Transformers</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> NA
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/crossq">CrossQ</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> - (continuous only)
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/s