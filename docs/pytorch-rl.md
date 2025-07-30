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

# TorchRL: Your Gateway to Advanced Reinforcement Learning with PyTorch

TorchRL provides a flexible, efficient, and modular framework for Reinforcement Learning (RL) research and development, built on PyTorch. **[Explore the TorchRL Repository](https://github.com/pytorch/rl) for comprehensive documentation, examples, and contributions.**

## Key Features

*   **Python-First Design:** Emphasizes Python for ease of use and flexibility.
*   **High Performance:** Optimized for demanding RL research applications, ensuring efficient training and evaluation.
*   **Modular and Extensible:** Highly modular architecture enabling easy customization and creation of new components.
*   **Well-Documented:** Thorough documentation with tutorials and API references to facilitate understanding and utilization.
*   **Rigorously Tested:** Ensures reliability and stability of the library.
*   **Reusable Functionals:** Offers a suite of reusable functions for costs, returns, and data processing.
*   **LLM API:** Integrated framework for post-training and fine-tuning of Language Models (LLMs).

    *   **Unified LLM Wrappers:** Hugging Face and vLLM support.
    *   **Conversation Management:** `History` class for multi-turn dialogue.
    *   **Tool Integration:** Support for Python code execution and function calls.
    *   **Specialized Objectives:** GRPO and SFT loss functions.
    *   **High-Performance Collectors:** Async data collection with distributed training.
    *   **Flexible Environments:** Transform-based architecture for reward computation.

## Why Choose TorchRL?

TorchRL is designed to align with the PyTorch ecosystem with minimal dependencies, making it easy to integrate with existing projects.  It provides reusable functions for cost functions, returns, and data processing and minimizes dependencies to reduce overhead.

## Getting Started

Learn the basics of TorchRL with these tutorials: [Getting Started Tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Core Concepts

TorchRL's design centers around key principles:

*   **Alignment with PyTorch:** Follows the structure and conventions of PyTorch libraries (e.g., datasets, transforms, models, data utilities).
*   **Minimal Dependencies:** Relies only on the Python standard library, NumPy, and PyTorch.

Read the [full paper](https://arxiv.org/abs/2306.00577) for a more curated description of the library.

## Documentation & Knowledge Base

*   **Documentation:** [TorchRL Documentation](https://pytorch.org/rl)
*   **RL Knowledge Base:** [RL Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html)
*   **Introductory Videos:**
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL has been used in a variety of research areas:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Simplifying RL with `TensorDict`

[TorchRL](https://github.com/pytorch/rl) solves this problem through [`TensorDict`](https://github.com/pytorch/tensordict/), a convenient data structure<sup>(1)</sup> that can be used to streamline one's
RL codebase.

## Features

- A common [interface for environments](https://github.com/pytorch/rl/blob/main/torchrl/envs)
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

- multiprocess and distributed [data collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py)<sup>(2)</sup>
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

- efficient<sup>(2)</sup> and generic<sup>(1)</sup> [replay buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py) with modularized storage:
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


- cross-library [environment transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py)<sup>(1)</sup>,
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

- various tools for distributed learning (e.g. [memory mapped tensors](https://github.com/pytorch/tensordict/blob/main/tensordict/memmap.py))<sup>(2)</sup>;
- various [architectures](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/) and models (e.g. [actor-critic](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/actors.py))<sup>(1)</sup>:
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

- exploration [wrappers](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py) and
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

- A series of efficient [loss modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives)
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

- a generic [trainer class](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py)<sup>(1)</sup> that
  executes the aforementioned training loop. Through a hooking mechanism,
  it also supports any logging or data transformation operation at any given
  time.

- various [recipes](https://github.com/pytorch/rl/blob/main/torchrl/trainers/helpers/models.py) to build models that
    correspond to the environment being deployed.

- **LLM API**: Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, 
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

If you feel a feature is missing from the library, please submit an issue!
If you would like to contribute to new features, check our [call for contributions](https://github.com/pytorch/rl/issues/509) and our [contribution](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) page.

## Examples, Tutorials, and Demos

Explore a variety of RL algorithms and implementations to get started quickly.

### Implemented Algorithms
*   DQN
*   DDPG
*   IQL
*   CQL
*   TD3
*   TD3+BC
*   A2C
*   PPO
*   SAC
*   REDQ
*   Dreamer v1
*   Decision Transformers
*   CrossQ
*   Gail
*   Impala
*   IQL (MARL)
*   DDPG (MARL)
*   PPO (MARL)
*   QMIX-VDN (MARL)
*   SAC (MARL)
*   RLHF
*   LLM API (GRPO)

### Compile Support, Tensordict-Free API, Modular Losses, and Continuous/Discrete environments
*   Refer to the original README for detailed support for each algorithm.

### Code Examples
*   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

### Tutorials and Demos
*   Get a sense of what the library can do, check out the [tutorials and demos](https://pytorch.org/rl/stable#tutorials)

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

Follow these steps to get TorchRL up and running:

### Virtual Environment Setup
```bash
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
```

Alternatively, use Conda:
```bash
conda create --name torchrl python=3.9
conda activate torchrl
```

### Install PyTorch

*   Instructions for installing PyTorch can be found [here](https://pytorch.org/get-started/locally/).

### Install TorchRL

*   Install the latest stable release:
    ```bash
    pip3 install torchrl
    ```
*   Build locally (for Windows or if a specific version is needed):
    ```bash
    # Install and build locally v0.8.1 of the library without cloning
    pip3 install git+https://github.com/pytorch/rl@v0.8.1
    # Clone the library and build it locally
    git clone https://github.com/pytorch/tensordict
    git clone https://github.com/pytorch/rl
    pip install -e tensordict
    pip install -e rl
    ```
    **Note:** Requires `cmake` via package manager, and `pip install "pybind11[global]"`.

*   Build wheels for distribution:
    ```bash
    pip install build
    python -m build --wheel
    ```
    Install with: `pip install torchrl<name>.whl`

*   Install the nightly build:
    ```bash
    pip3 install tensordict-nightly torchrl-nightly
    ```

### Optional Dependencies

Install additional libraries as needed:
```bash
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

## Troubleshooting

*   Refer to the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md) for solutions to common issues.

## Getting Help

*   Report bugs and ask questions via [this repo](https://github.com/pytorch/rl).
*   Discuss RL in PyTorch on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Contributions are welcome!  See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).  Check the [open contributions](https://github.com/pytorch/rl/issues/509) page. Use [pre-commit hooks](https://pre-commit.com/) to lint code before committing.

## Disclaimer

TorchRL is a beta feature and may have breaking changes.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.