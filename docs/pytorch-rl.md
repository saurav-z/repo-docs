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

# TorchRL: The PyTorch Library for Cutting-Edge Reinforcement Learning

**TorchRL** is an open-source library built on PyTorch, offering a comprehensive toolkit for reinforcement learning research and development.  [Explore the TorchRL repository](https://github.com/pytorch/rl) to get started!

## Key Features

*   **Python-First Design:** Prioritizes Python for ease of use, flexibility, and rapid prototyping.
*   **Efficient & Performant:** Optimized for performance, enabling complex RL applications.
*   **Modular and Customizable:**  A highly modular architecture allows for easy modification, extension, and component swapping.
*   **Well-Documented:** Thorough documentation with tutorials and an API reference ensures a smooth user experience.
*   **Rigorous Testing:** Comprehensive testing ensures reliability and stability.
*   **Reusable Functionals:** Includes a library of reusable functions for cost functions, returns, and data processing.
*   **Seamless Integration:**  Follows PyTorch conventions, making it easy to integrate with the PyTorch ecosystem.
*   **LLM API**: Complete framework for LLM fine-tuning with unified wrappers for Hugging Face and vLLM backends.

## What's New: LLM API - Revolutionizing Language Model Fine-tuning

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

- 🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
- 💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
- 🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
- 🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
- ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
- 🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

## Core Concepts

*   **TensorDict:** TorchRL leverages `TensorDict`, a flexible data structure that streamlines RL codebase development.  It allows for writing complex PPO training scripts in under 100 lines.  See the [TensorDict documentation](https://pytorch.github.io/tensordict/) for details.

    *   Seamlessly integrates with `torch.compile` and `functorch`.
    *   Supports tensor operations like stacking, concatenation, reshaping, indexing, and device/memory management.

*   **Environment Abstraction:**  Provides a standardized interface for diverse environments, including OpenAI Gym, DeepMind Control Suite, and model-based environments, along with batched environment containers for parallel execution.

    *   Environment transforms enhance data preprocessing.

*   **Data Collection:**  Supports multiprocess and distributed data collectors for efficient data gathering, with both synchronous and asynchronous modes.
*   **Replay Buffers:** Offers efficient, generic replay buffers with modularized storage options, including wrappers for offline RL datasets (e.g., D4RL).
*   **Modular Components:**  Provides a wide range of reusable modules, architectures, exploration wrappers, and loss functions to accelerate your research.

## Getting Started

Check our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) for quickly ramp up with the basic 
features of the library!

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Resources

*   **Comprehensive Documentation:** Find tutorials, API references, and detailed explanations at [https://pytorch.org/rl](https://pytorch.org/rl).
*   **RL Knowledge Base:** Explore the RL knowledge base to debug code and understand RL fundamentals: [https://pytorch.org/rl/stable/reference/knowledge_base.html](https://pytorch.org/rl/stable/reference/knowledge_base.html)
*   **Introductory Videos:** Check out helpful videos on [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl), the [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE), and [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share).

## Spotlight Publications

TorchRL's versatility is demonstrated in numerous research areas. Notable publications include:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing simplified and portable RL codebase with `TensorDict`

RL algorithms are very heterogeneous, and it can be hard to recycle a codebase
across settings (e.g. from online to offline, from state-based to pixel-based 
learning).
TorchRL solves this problem through [`TensorDict`](https://github.com/pytorch/tensordict/),
a convenient data structure<sup>(1)</sup> that can be used to streamline one's
RL codebase.
With this tool, one can write a *complete PPO training script in less than 100
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

Here is an example of how the [environment API](https://pytorch.org/rl/stable/reference/envs.html)
relies on tensordict to carry data from one function to another during a rollout
execution:
![Alt Text](https://github.com/pytorch/rl/blob/main/docs/source/_static/img/rollout.gif)

`TensorDict` makes it easy to re-use pieces of code across environments, models and
algorithms.
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

## Installation

### Prerequisites:
*   Python (>=3.7)
*   PyTorch (See instructions for install.)

### Installation Steps:

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv torchrl
    source torchrl/bin/activate  # On Windows use: venv\Scripts\activate
    ```
    Or:
    ```
    conda create --name torchrl python=3.9
    conda activate torchrl
    ```
2.  **Install PyTorch:**  Follow the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to install the appropriate PyTorch version for your system (CUDA, CPU, etc.).  Consider installing the nightly builds for cutting-edge features.
3.  **Install TorchRL:**

    *   **Stable Release:**
        ```bash
        pip3 install torchrl
        ```
    *   **Nightly Build (requires PyTorch nightly):**
        ```bash
        pip3 install tensordict-nightly torchrl-nightly
        ```
    *   **For local builds (potentially for Windows):**
        ```bash
        # Install and build locally v0.8.1 of the library without cloning
        pip3 install git+https://github.com/pytorch/rl@v0.8.1
        # Clone the library and build it locally
        git clone https://github.com/pytorch/tensordict
        git clone https://github.com/pytorch/rl
        pip install -e tensordict
        pip install -e rl
        ```
        Note that tensordict local build requires `cmake` to be installed via [homebrew](https://brew.sh/) (MacOS) or another package manager
        such as `apt`, `apt-get`, `conda` or `yum` but NOT `pip`, as well as `pip install "pybind11[global]"`.   
4.  **Optional Dependencies:** Install the necessary dependencies for specific features:
    ```bash
    pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher  # diverse
    pip3 install "moviepy<2.0.0" # rendering
    pip3 install dm_control # deepmind control suite
    pip3 install "gym[atari]" "gym[accept-rom-license]" pygame  # gym, atari games
    pip3 install pytest pyyaml pytest-instafail # tests
    pip3 install tensorboard # tensorboard
    pip3 install wandb # wandb
    ```
    See the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md) for troubleshooting.

## Examples, Tutorials, and Demo

A series of [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) are provided with an illustrative purpose:

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

Code examples and training scripts are also available:
*   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

## Contributing and Support

*   **Contribute:** TorchRL welcomes contributions!  See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for details.
*   **Report Issues:**  If you find a bug or have a suggestion, please open an issue in this repository.
*   **Ask Questions:**  For generic RL questions, use the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

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

## Disclaimer

TorchRL is a beta feature, and breaking changes may occur.  A deprecation period will be provided before significant changes.

## License

TorchRL is licensed under the MIT License.  See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.