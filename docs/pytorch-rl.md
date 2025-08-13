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

# TorchRL: Unleash the Power of Reinforcement Learning with PyTorch

[**Documentation**](#documentation-and-knowledge-base) | [**TensorDict**](#writing-simplified-and-portable-rl-codebase-with-tensordict) |
[**Features**](#features) | [**Examples, tutorials and demos**](#examples-tutorials-and-demos) | [**Citation**](#citation) | [**Installation**](#installation) |
[**Asking a question**](#asking-a-question) | [**Contributing**](#contributing)

TorchRL is an open-source, Python-first library that empowers researchers and developers to build and experiment with Reinforcement Learning (RL) models using the flexibility and efficiency of PyTorch.  Get started with the original repo [here](https://github.com/pytorch/rl).

## Key Features:

*   **Python-First Design:** Designed for ease of use and flexibility, built with Python as the primary language.
*   **Efficient Performance:** Optimized for demanding RL research applications.
*   **Modular and Customizable:** Highly modular architecture allowing easy swapping, transformation, or creation of new components.
*   **Well-Documented:** Thorough documentation ensures that users can quickly understand and utilize the library.
*   **Rigorously Tested:** Ensures reliability and stability.
*   **Reusable Functionals:** Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   **LLM API**: Comprehensive framework for Language Model Fine-tuning:
    *   Unified LLM Wrappers (Hugging Face, vLLM).
    *   Conversation Management (History class).
    *   Tool Integration (Python code execution, function calling).
    *   Specialized Objectives (GRPO, SFT).
    *   High-Performance Collectors (async data collection).
    *   Flexible Environments (transform-based architecture).

### Design Principles

*   **Aligns with PyTorch Ecosystem:** Follows the structure and conventions of popular PyTorch libraries.
*   **Minimal Dependencies:** Requires only the Python standard library, NumPy, and PyTorch.

Read the [full paper](https://arxiv.org/abs/2306.00577) for a more curated description of the library.

## 🚀 What's New: LLM API

TorchRL now features a comprehensive **LLM API** for post-training and fine-tuning language models!

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
Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

## Getting Started

Check our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) for quickly ramp up with the basic 
features of the library!

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

Find comprehensive documentation, tutorials, and the API reference [here](https://pytorch.org/rl).

Also, you can debug your code or simply learn the basics of RL with the RL knowledge base [here](https://pytorch.org/rl/stable/reference/knowledge_base.html).

Check out these introductory videos to get to know the library better:

-   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
-   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
-   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is versatile and can be used in various fields. Here are a few examples:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL simplifies RL codebase development through `TensorDict`, enabling code reuse across different RL settings.

With this tool, one can write a *complete PPO training script in less than 100 lines of code*!

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

`TensorDict` abstracts input/output signatures, enabling code reuse for environments, models, collectors, and losses.

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

  <details>
    <summary>Code</summary>

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

## Features

*   [Environment Interface](https://github.com/pytorch/rl/blob/main/torchrl/envs) supporting common libraries, stateless execution, and batched environments.
*   Multiprocess and distributed [Data Collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py).
*   Efficient and generic [Replay Buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py).
*   Cross-library [Environment Transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py).
*   Tools for distributed learning and various [Architectures/Models](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/).
*   Exploration [Wrappers/Modules](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py).
*   Efficient [Loss Modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives) and vectorized functional computations.
*   Generic [Trainer Class](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py).
*   Various [recipes](https://github.com/pytorch/rl/blob/main/torchrl/trainers/helpers/models.py) to build models.

If you spot a feature is missing, please submit an issue!
If you would like to contribute to new features, check our [call for contributions](https://github.com/pytorch/rl/issues/509) and our [contribution](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) page.

## Examples, Tutorials, and Demos

A series of [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/) are provided:

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

Check the [examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) directory for more details 
about handling the various configuration settings.

We also provide [tutorials and demos](https://pytorch.org/rl/stable#tutorials) that give a sense of
what the library can do.

*   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
*   [RLHF](examples/rlhf)
*   [Memory-mapped replay buffers](examples/torchrl_features)

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

Depending on the use of torchrl that you want to make, you may want to 
install the latest (nightly) PyTorch release or the latest stable version of PyTorch.
See [here](https://pytorch.org/get-started/locally/) for a detailed list of commands, 
including `pip3` or other special installation instructions.

TorchRL offers a few pre-defined dependencies such as `"torchrl[tests]"`, `"torchrl[atari]"` etc. 

#### Torchrl

You can install the **latest stable release** by using
```bash
pip3 install torchrl
```
This should work on linux (including AArch64 machines), Windows 10 and OsX (Metal chips only).
On certain Windows machines (Windows 11), one should build the library locally.
This can be done in two ways:

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

One can also build the wheels to distribute to co-workers using
```bash
pip install build
python -m build --wheel
```
Your wheels will be stored there `./dist/torchrl<name>.whl` and installable via
```bash
pip install torchrl<name>.whl
```

The **nightly build** can be installed via
```bash
pip3 install tensordict-nightly torchrl-nightly
```
which we currently only ship for Linux machines.
Importantly, the nightly builds require the nightly builds of PyTorch too.
Also, a local build of torchrl with the nightly build of tensordict may fail - install both nightlies or both local builds but do not mix them.


**Disclaimer**: As of today, TorchRL is roughly compatible with any pytorch version >= 2.1 and installing it will not
directly require a newer version of pytorch to be installed. Indirectly though, tensordict still requires the latest
PyTorch to be installed and we are working hard to loosen that requirement. 
The C++ binaries of TorchRL (mainly for prioritized replay buffers) will only work with PyTorch 2.7.0 and above.
Some features (e.g., working with nested jagged tensors) may also
be limited with older versions of pytorch. It is recommended to use the latest TorchRL with the latest PyTorch version
unless there is a strong reason not to do so.

**Optional dependencies**

The following libraries can be installed depending on the usage one wants to
make of torchrl:
```
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

Versioning issues can cause error message of the type ```undefined symbol```
and such. For these, refer to the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md)
for a complete explanation and proposed workarounds.

## Asking a Question

If you spot a bug in the library, please raise an issue in this repo.

For generic RL questions, post on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).

## Contributing

Collaborations are welcome! Fork, submit issues, and create PRs.
Check the detailed contribution guide [here](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md).
A list of open contributions can be found [here](https://github.com/pytorch/rl/issues/509).

Install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`) to check linting issues. Disable with `-n` in commit command.

## Disclaimer

This library is released as a PyTorch beta feature.
BC-breaking changes are likely to happen but they will be introduced with a deprecation
warranty after a few release cycles.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.