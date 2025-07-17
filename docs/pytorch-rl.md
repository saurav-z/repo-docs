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

# TorchRL: Your PyTorch Toolkit for Cutting-Edge Reinforcement Learning

**TorchRL is an open-source library that empowers researchers and developers to build efficient and modular reinforcement learning (RL) solutions with PyTorch.** Explore the [original repository](https://github.com/pytorch/rl) for more details.

## Key Features

*   **Python-first Design:** Built with Python for ease of use and flexibility in RL development.
*   **Efficient and Optimized:** Engineered for high performance, supporting demanding RL research applications.
*   **Modular and Extensible:** Highly modular architecture enabling easy customization, component swapping, and expansion.
*   **Comprehensive Documentation:** Thoroughly documented to facilitate quick understanding and utilization of the library.
*   **Rigorous Testing:** Ensures reliability and stability through comprehensive testing.
*   **Reusable Functionals:** Provides a set of highly reusable functions for cost functions, returns, and data processing.
*   **LLM API:**  A complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends, supporting RLHF, supervised fine-tuning, and tool-augmented training.

## What's New: LLM API 🚀

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models!

*   🤖 **Unified LLM Wrappers**: Seamless integration with Hugging Face models and vLLM inference engines - more to come!
*   💬 **Conversation Management**: Advanced [`History`](torchrl/data/llm/history.py) class for multi-turn dialogue with automatic chat template detection
*   🛠️ **Tool Integration**: [Built-in support](torchrl/envs/llm/transforms/) for Python code execution, function calling, and custom tool transforms
*   🎯 **Specialized Objectives**: [GRPO](torchrl/objectives/llm/grpo.py) (Group Relative Policy Optimization) and [SFT](torchrl/objectives/llm/sft.py) loss functions optimized for language models
*   ⚡ **High-Performance Collectors**: [Async data collection](torchrl/collectors/llm/) with distributed training support
*   🔄 **Flexible Environments**: Transform-based architecture for reward computation, data loading, and conversation augmentation

Check out the [complete documentation](https://pytorch.org/rl/main/reference/llms.html) and [GRPO implementation example](https://github.com/pytorch/rl/tree/main/sota-implementations/grpo) to get started!

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

Dive into RL with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

Access detailed documentation and tutorials: [Documentation](https://pytorch.org/rl). Explore our RL knowledge base to debug code and learn the basics: [Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html).

Introductory videos:

-   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
-   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
-   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL's versatility shines in various fields. Notable publications include:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

## Streamlined RL Code with `TensorDict`

`TensorDict` simplifies RL codebases and enables portability across settings.  You can write a complete PPO training script in less than 100 lines of code using TensorDict. Explore the [TensorDict documentation](https://pytorch.github.io/tensordict/) for details.

Here is an example of how the [environment API](https://pytorch.org/rl/stable/reference/envs.html)
relies on tensordict to carry data from one function to another during a rollout
execution:
![Alt Text](https://github.com/pytorch/rl/blob/main/docs/source/_static/img/rollout.gif)

## Features in Detail

*   **Environments**:  A common [interface for environments](https://github.com/pytorch/rl/blob/main/torchrl/envs), with batched environments and tensor specifications. Check the [documentation](https://pytorch.org/rl/stable/reference/envs.html)
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

*   **Data Collectors**:  Multiprocess and distributed [data collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py) that work synchronously or asynchronously.
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

*   **Replay Buffers**: Efficient and generic [replay buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py) with modularized storage. Replay buffers are also offered as wrappers around common datasets for *offline RL*.
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

*   **Environment Transforms**:  Cross-library [environment transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py), executed on device and in a vectorized fashion.
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

*   **Architectures and Models**: Various [architectures](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/) and models (e.g. [actor-critic](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/actors.py)).
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

*   **Exploration Wrappers and Modules**: Exploration [wrappers](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py) and
    [modules](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/exploration.py) to easily swap between exploration and exploitation.
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

*   **Loss Modules and Value Computation**: Efficient [loss modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives)
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

*   **Trainer Class**: A generic [trainer class](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py) that executes the training loop.

*   **Recipes**: Various [recipes](https://github.com/pytorch/rl/blob/main/torchrl/trainers/helpers/models.py) to build models that
    correspond to the environment being deployed.

## Examples, Tutorials, and Demos

Explore [State-of-the-Art implementations](https://github.com/pytorch/rl/blob/main/sota-implementations/):

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

[Code examples](examples/) and [tutorials](https://pytorch.org/rl/stable#tutorials) are available.

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

### Prerequisites

*   Python (>= 3.8)
*   PyTorch (Install the latest stable or nightly release)
*   [Install tensordict](https://pytorch.github.io/tensordict/)

### Installation Steps
```bash
# Create a new virtual environment:
python -m venv torchrl
source torchrl/bin/activate  # On Windows use: venv\Scripts\activate

# Or create a conda environment
conda create --name torchrl python=3.9
conda activate torchrl

# Install the latest stable release by using:
pip3 install torchrl

# For nightly build
pip3 install tensordict-nightly torchrl-nightly

# Optional dependencies can be installed for extended use:
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher "moviepy<2.0.0" dm_control "gym[atari]" "gym[accept-rom-license]" pygame pytest pyyaml pytest-instafail tensorboard wandb
```

Refer to the [versioning issues document](https://github.com/pytorch/rl/blob/main/knowledge_base/VERSIONING_ISSUES.md) for troubleshooting.

## Get Involved

*   **Report Bugs:**  Raise issues in this repository.
*   **Ask Questions:**  Post questions on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).
*   **Contribute:**  Fork the repository, submit issues, and create pull requests. See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for details.

## Disclaimer

TorchRL is a PyTorch beta feature; expect BC-breaking changes with a deprecation warranty.

## License

TorchRL is licensed under the MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.