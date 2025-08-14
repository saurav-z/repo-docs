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

TorchRL is an open-source library designed to empower researchers and developers to build, train, and deploy reinforcement learning models efficiently, offering a flexible and modular framework for tackling complex RL challenges.  Check out the [original repository](https://github.com/pytorch/rl) for more details.

<p align="center">
  <img src="docs/source/_static/img/icon.png"  width="200" >
</p>

[**Documentation**](#documentation-and-knowledge-base) | [**TensorDict**](#writing-simplified-and-portable-rl-codebase-with-tensordict) |
[**Features**](#features) | [**Examples, tutorials and demos**](#examples-tutorials-and-demos) | [**Citation**](#citation) | [**Installation**](#installation) |
[**Asking a question**](#asking-a-question) | [**Contributing**](#contributing)

## Key Features

*   **Python-First Design:** Prioritizes Python for ease of use, flexibility, and a smooth development experience.
*   **High Performance:** Optimized for demanding RL research applications, enabling efficient training and evaluation.
*   **Modular & Customizable:** Offers a highly modular architecture that allows for easy swapping, transformation, or creation of new components.
*   **Comprehensive Documentation:** Provides thorough documentation, tutorials, and API references to accelerate your learning and development.
*   **Rigorous Testing:** Ensures reliability and stability through rigorous testing and continuous integration.
*   **Reusable Functionals:** Includes a rich set of reusable functions for cost functions, returns, and data processing, streamlining your workflow.
*   **Seamless PyTorch Integration:** Aligns with the PyTorch ecosystem, following established structures and conventions for a familiar and intuitive experience.
*   **Minimal Dependencies:** Keeps dependencies to a minimum, requiring only Python, NumPy, and PyTorch, with optional dependencies for common environment libraries and datasets.
*   **LLM API**: Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends,  conversation management with automatic chat template detection, tool integration (Python execution, function calling),  specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning,  and tool-augmented training scenarios.

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

## Writing Simplified and Portable RL Codebase with `TensorDict`

TorchRL simplifies RL code through `TensorDict`, a powerful data structure for streamlining your RL codebase.  With `TensorDict`, you can write a complete PPO training script in a compact and efficient manner.

(Further text from original README, focusing on the benefits of TensorDict)

## Getting Started

Check our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started) for quickly ramp up with the basic  features of the library!

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

## Documentation and Knowledge Base

*   The TorchRL documentation is available [here](https://pytorch.org/rl), providing tutorials and an API reference.
*   A RL knowledge base is provided [here](https://pytorch.org/rl/stable/reference/knowledge_base.html) to help you debug your code, or simply learn the basics of RL.
*   Introductory videos are available:
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

## Spotlight Publications

TorchRL is domain-agnostic and used across many different fields:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

(Rest of original README - Features, Examples, Citation, Installation, Asking a Question, Contributing, Disclaimer, License - included)

## Features

*   A common [interface for environments](https://github.com/pytorch/rl/blob/main/torchrl/envs) which supports common libraries (OpenAI gym, deepmind control lab, etc.)<sup>(1)</sup> and state-less execution (e.g. Model-based environments). The [batched environments](https://github.com/pytorch/rl/blob/main/torchrl/envs/batched_envs.py) containers allow parallel execution<sup>(2)</sup>. A common PyTorch-first class of [tensor-specification class](https://github.com/pytorch/rl/blob/main/torchrl/data/tensor_specs.py) is also provided. TorchRL's environments API is simple but stringent and specific. Check the [documentation](https://pytorch.org/rl/stable/reference/envs.html) and [tutorial](https://pytorch.org/rl/stable/tutorials/pendulum.html) to learn more!
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

*   multiprocess and distributed [data collectors](https://github.com/pytorch/rl/blob/main/torchrl/collectors/collectors.py)<sup>(2)</sup> that work synchronously or asynchronously. Through the use of TensorDict, TorchRL's training loops are made very similar to regular training loops in supervised learning (although the "dataloader" -- read data collector -- is modified on-the-fly):
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

*   efficient<sup>(2)</sup> and generic<sup>(1)</sup> [replay buffers](https://github.com/pytorch/rl/blob/main/torchrl/data/replay_buffers/replay_buffers.py) with modularized storage:
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


*   cross-library [environment transforms](https://github.com/pytorch/rl/blob/main/torchrl/envs/transforms/transforms.py)<sup>(1)</sup>, executed on device and in a vectorized fashion<sup>(2)</sup>, which process and prepare the data coming out of the environments to be used by the agent:
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
    Other transforms include: reward scaling (`RewardScaling`), shape operations (concatenation of tensors, unsqueezing etc.), concatenation of successive operations (`CatFrames`), resizing (`Resize`) and many more.

    Unlike other libraries, the transforms are stacked as a list (and not wrapped in each other), which makes it easy to add and remove them at will:
    ```python
    env.insert_transform(0, NoopResetEnv())  # inserts the NoopResetEnv transform at the index 0
    ```
    Nevertheless, transforms can access and execute operations on the parent environment:
    ```python
    transform = env.transform[1]  # gathers the second transform of the list
    parent_env = transform.parent  # returns the base environment of the second transform, i.e. the base env + the first transform
    ```
    </details>

*   various tools for distributed learning (e.g. [memory mapped tensors](https://github.com/pytorch/tensordict/blob/main/tensordict/memmap.py))<sup>(2)</sup>;
*   various [architectures](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/) and models (e.g. [actor-critic](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/actors.py))<sup>(1)</sup>:
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

*   exploration [wrappers](https://github.com/pytorch/rl/blob/main/torchrl/modules/tensordict_module/exploration.py) and [modules](https://github.com/pytorch/rl/blob/main/torchrl/modules/models/exploration.py) to easily swap between exploration and exploitation<sup>(1)</sup>:
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

*   A series of efficient [loss modules](https://github.com/pytorch/rl/tree/main/torchrl/objectives) and highly vectorized [functional return and advantage](https://github.com/pytorch/rl/blob/main/torchrl/objectives/value/functional.py) computation.

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

*   a generic [trainer class](https://github.com/pytorch/rl/blob/main/torchrl/trainers/trainers.py)<sup>(1)</sup> that executes the aforementioned training loop. Through a hooking mechanism, it also supports any logging or data transformation operation at any given time.

*   various [recipes](https://github.com/pytorch/rl/blob/main/torchrl/trainers/helpers/models.py) to build models that correspond to the environment being deployed.

*   **LLM API**: Complete framework for language model fine-tuning with unified wrappers for Hugging Face and vLLM backends,  conversation management with automatic chat template detection, tool integration (Python execution, function calling),  specialized objectives (GRPO, SFT), and high-performance async collectors. Perfect for RLHF, supervised fine-tuning,  and tool-augmented training scenarios.
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

If you feel a feature is missing from the library, please submit an issue! If you would like to contribute to new features, check our [call for contributions](https://github.com/pytorch/rl/issues/509) and our [contribution](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) page.


## Examples, tutorials and demos

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

[Code examples](examples/) displaying toy code snippets and training scripts are also available
- [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
- [RLHF](examples/rlhf)
- [Memory-mapped replay buffers](examples/torchrl_features)


Check the [examples](https://github.com/pytorch/rl/blob/main/sota-implementations/) directory for more details about handling the various configuration settings.

We also provide [tutorials and demos](https://pytorch.org/rl/stable#tutorials) that give a sense of what the library can do.

## Citation

If you're using TorchRL, please refer to this BibTeX entry to cite this work:
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

Depending on the use of torchrl that you want to make, you may want to install the latest (nightly) PyTorch release or the latest stable version of PyTorch.
See [here](https://pytorch.org/get-started/locally/) for a detailed list of commands, including `pip3` or other special installation instructions.

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

Note that tensordict local build requires `cmake` to be installed via [homebrew](https://brew.sh/) (MacOS) or another package manager such as `apt`, `apt-get`, `conda` or `yum` but NOT `pip`, as well as `pip install "pybind11[global]"`.

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


**Disclaimer**: As of today, TorchRL is roughly compatible with any pytorch version >= 2.1 and installing it will not directly require a newer version of pytorch to be installed. Indirectly though,