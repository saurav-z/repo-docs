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

# TorchRL: Your Gateway to Cutting-Edge Reinforcement Learning in PyTorch

[**TorchRL**](https://github.com/pytorch/rl) is an open-source, PyTorch-based library designed to empower researchers and developers with a flexible and efficient toolkit for Reinforcement Learning (RL).

## Key Features

*   🐍 **Python-first Design:** Prioritizes Python for ease of use and flexibility.
*   ⏱️ **High Performance:** Optimized for speed, supporting demanding RL research.
*   🧮 **Modular & Customizable:** A modular architecture makes components easy to swap, modify, or create.
*   📚 **Comprehensive Documentation:** Thorough documentation accelerates understanding and utilization.
*   ✅ **Rigorously Tested:** Ensures reliability and stability through extensive testing.
*   ⚙️ **Reusable Functionals:** Offers a library of reusable functions for cost calculations, returns, and data handling.
*   🤖 **LLM API**: A complete framework for LLM fine-tuning, including wrappers, conversation management, tool integration, specialized objectives (GRPO, SFT), and high-performance collectors, ideal for RLHF and tool-augmented training.

## What's New

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

## Getting Started

Explore the basic features with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) to quickly start using the library.

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

Access detailed documentation, tutorials, and the API reference [here](https://pytorch.org/rl). Also, find a helpful RL knowledge base [here](https://pytorch.org/rl/stable/reference/knowledge_base.html) to troubleshoot your code and learn the fundamentals.

## Spotlight Publications

TorchRL has been utilized across multiple domains. Here are a few examples:

-   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
-   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
-   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
-   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
-   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
-   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL simplifies RL codebase with [`TensorDict`](https://github.com/pytorch/tensordict/), a data structure that streamlines RL development.  Write complete PPO training scripts in under 100 lines!

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

`TensorDict` provides a flexible data structure for simplifying the code.

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

Using this, TorchRL abstracts away the input / output signatures of the modules, env, 
collectors, replay buffers and losses of the library, allowing all primitives
to be easily recycled across settings.

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

TensorDict comes with a dedicated [`tensordict.nn`](https://pytorch.github.io/tensordict/reference/nn.html)
module that contains everything you might need to write your model with it.
And it is `functorch` and `torch.compile` compatible!

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

## Features (Detailed)

*   **Environments:** Unified interface for environments, including popular libraries and state-less execution. Batched environments enable parallel execution. Tensor-specification class provided.
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

*   **Data Collectors:** Multiprocess and distributed data collectors, both synchronous and asynchronous, and support for RLHF.
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

*   **Replay Buffers:** Efficient and generic replay buffers with modular storage, including memory-mapped options. Offline RL support via wrappers around common datasets.
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

*   **Environment Transforms:** Cross-library environment transforms executed on-device and vectorized for data processing.
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

*   **Models and Architectures:** Various models and architectures, including actor-critic and exploration wrappers.
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

*   **Exploration Wrappers:** Exploration wrappers and modules for easy switching between exploration and exploitation.
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

*   **Loss Modules and Utilities:** Efficient loss modules and vectorized functional return and advantage computation.
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

*   **Trainer Class:** A generic trainer class for the training loop, with hooking mechanisms for logging and data transformation.

*   **Recipes:** Recipes to build models for specific environments.

## Examples, Tutorials, and Demos

Explore a variety of State-of-the-Art implementations and detailed [code examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) covering various RL algorithms.

**Key Examples:**
*   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

Find more details about handling the various configuration settings in the [examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) directory. Also, check out the [tutorials and demos](https://pytorch.org/rl/stable#tutorials) to get a better understanding.

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

### Create a Virtual Environment:

```bash
python -m venv torchrl
source torchrl/bin/activate  # Or venv\Scripts\activate on Windows
```

### Install Dependencies:

*   **PyTorch:** Install the appropriate version of PyTorch (stable or nightly) based on your needs.  See [PyTorch Installation](https://pytorch.org/get-started/locally/) for detailed instructions.
*   **TorchRL:**
    ```bash
    pip3 install torchrl  # For the latest stable release
    ```
    or to install the local library:
    ```bash
    # Install and build locally v0.8.1 of the library without cloning
    pip3 install git+https://github.com/pytorch/rl@v0.8.1
    # Clone the library and build it locally
    git clone https://github.com/pytorch/tensordict
    git clone https://github.com/pytorch/rl
    pip install -e tensordict
    pip install -e rl
    ```
    or the nightly version:
    ```bash
    pip3 install tensordict-nightly torchrl-nightly
    ```

    **Note:** Nightly builds require PyTorch nightly builds.

*   **Optional Dependencies:** Install additional libraries as needed for your specific use cases: `tqdm`, `tensorboard`, `moviepy`, `dm_control`, `gym`, `atari`, `pytest`, `pyyaml`, `wandb`.

## Asking a Question

For bug reports, create an issue in this repository. For general questions, use the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome!  See the [CONTRIBUTING.md](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) guide for details. Check the open contributions in [here](https://github.com/pytorch/rl/issues/509).

## Disclaimer

TorchRL is a PyTorch beta feature, with possible BC-breaking changes introduced after a deprecation warning.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.