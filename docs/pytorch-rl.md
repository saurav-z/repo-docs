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

TorchRL is an open-source, PyTorch-based library designed to streamline Reinforcement Learning (RL) research and development, offering a modular and efficient framework. Explore the original repository on [GitHub](https://github.com/pytorch/rl).

## Key Features

*   **Python-First Design:** Designed for ease of use and flexibility, prioritizing Python as the primary language.
*   **Efficient Performance:** Optimized for demanding RL applications, ensuring speed and scalability.
*   **Modular and Extensible:** Highly modular architecture enables easy customization, transformation, and creation of new components.
*   **Comprehensive Documentation:** Thorough documentation, including tutorials and API reference, facilitates quick understanding and utilization.
*   **Robust Testing:** Rigorously tested to guarantee reliability and stability.
*   **Reusable Functionals:** Provides a rich set of reusable functions for cost functions, returns, and data processing.
*   **LLM API:** A complete framework for language model fine-tuning, perfect for RLHF, supervised fine-tuning, and tool-augmented training.

    *   **Unified LLM Wrappers**: Integrates with Hugging Face and vLLM backends.
    *   **Conversation Management**: Includes the `History` class for multi-turn dialogues.
    *   **Tool Integration**: Support for Python code execution and custom tool transforms.
    *   **Specialized Objectives**: Incorporates GRPO and SFT loss functions.
    *   **High-Performance Collectors**: Supports async data collection and distributed training.
    *   **Flexible Environments**: Transform-based architecture for reward computation and conversation augmentation.

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

## Design Principles

*   **Alignment with PyTorch Ecosystem:** Adheres to the structure and conventions of popular PyTorch libraries.
*   **Minimal Dependencies:** Relies primarily on the Python standard library, NumPy, and PyTorch.  Includes optional dependencies for common environment libraries (e.g., OpenAI Gym) and datasets (D4RL, OpenX...).

Read the [full paper](https://arxiv.org/abs/2306.00577) for a more curated description of the library.

## Getting Started

Begin your RL journey with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) for an immediate grasp of the library's foundational features!

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

Access the comprehensive TorchRL documentation [here](https://pytorch.org/rl), including tutorials and API references. Also, explore the RL knowledge base [here](https://pytorch.org/rl/stable/reference/knowledge_base.html) to enhance your RL understanding.

Learn more with these introductory videos:
- [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
- [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
- [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is a versatile tool applicable across various domains. Here are some prominent examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
  Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL simplifies RL codebase creation through `TensorDict` by offering a convenient data structure for streamlining RL coding tasks. With this tool, you can write a *complete PPO training script in less than 100 lines of code*!

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

The [environment API](https://pytorch.org/rl/stable/reference/envs.html)
relies on tensordict to carry data from one function to another during a rollout
execution:
![Alt Text](https://github.com/pytorch/rl/blob/main/docs/source/_static/img/rollout.gif)

`TensorDict` facilitates code reuse across diverse environments, models, and algorithms.
  <details>
    <summary>Code</summary>
  
  For instance, here's how to code a rollout in TorchRL:

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

TorchRL abstracts input/output signatures for modules, environments, collectors, replay buffers, and losses.
  <details>
    <summary>Code</summary>

  Here's another example of an off-policy training loop in TorchRL (assuming 
  that a data collector, a replay buffer, a loss and an optimizer have been instantiated):
  
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
  This training loop can be re-used across algorithms as it makes a minimal number of assumptions about the structure of the data.
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

TensorDict offers a dedicated [`tensordict.nn`](https://pytorch.github.io/tensordict/reference/nn.html)
module for model development and is `functorch` and `torch.compile` compatible!

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

  The `TensorDictSequential` class allows to branch sequences of `nn.Module` instances in a highly modular way.
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

  Check [TensorDict tutorials](https://pytorch.github.io/tensordict/) to
  learn more!

## Features in Detail

*   **Environment Interface:** A unified interface for environments, supporting common libraries, state-less execution, and batched environments for parallel execution. The package provides tensor-specification classes to simplify data handling. Explore the [documentation](https://pytorch.org/rl/stable/reference/envs.html) and [tutorial](https://pytorch.org/rl/stable/tutorials/pendulum.html) to get started!
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

*   **Data Collectors:** Multi-process and distributed data collectors, supporting synchronous and asynchronous operations.
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

*   **Replay Buffers:** Efficient and generic replay buffers with modularized storage.
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

*   **Environment Transforms:** Cross-library environment transforms, executed on device and in a vectorized fashion, for data processing and preparation.
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

*   **Additional Features**:
    *   Tools for distributed learning, including memory-mapped tensors.
    *   Various architectures and models (e.g., actor-critic).
    *   Exploration wrappers and modules for easy exploration/exploitation switching.
    *   Efficient loss modules and vectorized functional return and advantage computation.
    *   A generic trainer class for training loops and data transformation.
    *   Various recipes for building models corresponding to the environment.

*   **LLM API**: Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, conversation management with automatic chat template detection, tool integration (Python execution, function calling), specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning, and tool-augmented training scenarios.
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

If you feel a feature is missing from the library, please submit an issue!
If you would like to contribute to new features, check our [call for contributions](https://github.com/pytorch/rl/issues/509) and our [contribution](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) page.

## Examples, Tutorials, and Demos

Explore a range of implementations for various algorithms:

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
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/gail">Gail</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> NA
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/impala">Impala</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/iql.py">IQL (MARL)</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/maddpg_iddpg.py">DDPG (MARL)</a>
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
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/mappo_ippo.py">PPO (MARL)</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/qmix_vdn.py">QMIX-VDN (MARL)</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> NA
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/multiagent/sac.py">SAC (MARL)</a>
   </td>
   <td> untested
   </td>
   <td> +
   </td>
   <td> -
   </td>
   <td> +
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/examples/rlhf">RLHF</a>
   </td>
   <td> NA
   </td>
   <td> +
   </td>
   <td> NA
   </td>
   <td> NA
   </td>
  </tr>
  <tr>
   <td><a href="https://github.com/pytorch/rl/blob/main/sota-implementations/grpo">LLM API (GRPO)</a>
   </td>
   <td> NA
   </td>
   <td> +
   </td>
   <td> +
   </td>
   <td> NA
   </td>
  </tr>
</table>

** The number indicates expected speed-up compared to eager mode when executed on CPU. Numbers may vary depending on
  architecture and device.

and many more to come!

[Code examples](examples/) displaying toy code snippets and training scripts are also available 
- [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
- [RLHF](examples/rlhf)
- [Memory-mapped replay buffers](examples/torchrl