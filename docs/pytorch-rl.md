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

[**TorchRL**](https://github.com/pytorch/rl) is an open-source library designed to empower researchers and developers with flexible, efficient, and modular tools for Reinforcement Learning (RL) using PyTorch.

<p align="center">
  <img src="docs/source/_static/img/icon.png"  width="200" >
</p>

**Key Features:**

*   🐍 **Python-first Design:** Prioritizes Python for ease of use and flexibility.
*   ⏱️ **High Performance:** Optimized for speed, essential for demanding RL applications.
*   🧮 **Modular and Extensible:** Modular architecture enables easy customization and integration of components.
*   📚 **Comprehensive Documentation:** Detailed documentation for quick understanding and utilization.
*   ✅ **Reliable and Stable:** Rigorously tested for dependable performance.
*   ⚙️ **Reusable Components:** Provides reusable functions for common RL tasks (cost functions, returns, data processing).
*   🧠 **LLM API:** A complete framework for language model fine-tuning.

**What's New: LLM API**

TorchRL now includes a comprehensive **LLM API** for post-training and fine-tuning of language models! This new framework provides everything you need for RLHF, supervised fine-tuning, and tool-augmented training:

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

**Getting Started**

Quickly learn the basics with our [Getting Started tutorials](https://pytorch.org/rl/stable/index.html#getting-started).

<p align="center">
  <img src="docs/ppo.png"  width="800" >
</p>

**Documentation & Knowledge Base**

*   Comprehensive [Documentation](https://pytorch.org/rl) with tutorials and API reference.
*   RL [Knowledge Base](https://pytorch.org/rl/stable/reference/knowledge_base.html) for troubleshooting and learning RL fundamentals.
*   Introductory videos:
    *   [TalkRL podcast](https://www.talkrl.com/episodes/vincent-moens-on-torchrl)
    *   [TorchRL intro at PyTorch day 2022](https://youtu.be/cIKMhZoykEE)
    *   [PyTorch 2.0 Q&A: TorchRL](https://www.youtube.com/live/myEfUoYrbts?feature=share)

**Key Design Principles**

*   🔥 **PyTorch Ecosystem Alignment:** Follows PyTorch conventions for seamless integration.
*   ➖ **Minimal Dependencies:** Relies primarily on the Python standard library, NumPy, and PyTorch, with optional dependencies for common environments and datasets.

**Writing Simplified and Portable RL Codebase with `TensorDict`**

TorchRL utilizes [`TensorDict`](https://github.com/pytorch/tensordict/), a convenient data structure for streamlining RL code.  This allows for a complete PPO training script in under 100 lines of code!  Learn more in the [TensorDict tutorials](https://pytorch.github.io/tensordict/).

**Spotlight Publications**

TorchRL is being used in a variety of fields:

*   [ACEGEN](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00895): Reinforcement Learning of Generative Chemical Agents
    for Drug Discovery
*   [BenchMARL](https://www.jmlr.org/papers/v25/23-1612.html): Benchmarking Multi-Agent Reinforcement Learning
*   [BricksRL](https://arxiv.org/abs/2406.17490): A Platform for Democratizing Robotics and Reinforcement Learning
    Research and Education with LEGO
*   [OmniDrones](https://ieeexplore.ieee.org/abstract/document/10409589): An Efficient and Flexible Platform for Reinforcement Learning in Drone Control
*   [RL4CO](https://arxiv.org/abs/2306.17100): an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark
*   [Robohive](https://proceedings.neurips.cc/paper_files/paper/2023/file/8a84a4341c375b8441b36836bb343d4e-Paper-Datasets_and_Benchmarks.pdf): A unified framework for robot learning

**Features**

*   **Environment Interface:** A common interface supporting various environments (OpenAI Gym, DeepMind Control, etc.) and state-less execution.  Includes batched environments for parallel execution and tensor specifications.
*   **Data Collectors:** Multiprocess and distributed data collectors for synchronous and asynchronous data gathering.
*   **Replay Buffers:** Efficient and generic replay buffers with modularized storage.  Also offered as wrappers around common datasets for offline RL.
*   **Environment Transforms:** Cross-library transforms executed on-device and in a vectorized manner, to pre-process environment data.
*   **Model Architectures and Models:** Various built-in architectures and models, like actor-critic.
*   **Exploration Wrappers and Modules:** Easy swapping between exploration and exploitation strategies.
*   **Loss Modules & Functional Computation:** A series of efficient loss modules and vectorized functional return and advantage computation.
*   **Trainer Class:**  A generic trainer class that executes the training loop.  Supports logging and data transformation.
*   **Recipes:** Recipes to build models that correspond to the environment being deployed.

**Examples, Tutorials, and Demos**

*   **State-of-the-Art Implementations:**
    *   [DQN](sota-implementations/dqn)
    *   [DDPG](sota-implementations/ddpg/ddpg.py)
    *   [IQL](sota-implementations/iql/)
    *   [CQL](sota-implementations/cql/cql_offline.py)
    *   [TD3](sota-implementations/td3/td3.py)
    *   [TD3+BC](sota-implementations/td3_bc/td3_bc.py)
    *   [A2C](examples/a2c/)
    *   [PPO](sota-implementations/ppo/)
    *   [SAC](sota-implementations/sac/sac.py)
    *   [REDQ](sota-implementations/redq/redq.py)
    *   [Dreamer v1](sota-implementations/dreamer/dreamer.py)
    *   [Decision Transformers](sota-implementations/decision_transformer)
    *   [CrossQ](sota-implementations/crossq)
    *   [Gail](sota-implementations/gail)
    *   [Impala](sota-implementations/impala)
    *   [IQL (MARL)](sota-implementations/multiagent/iql.py)
    *   [DDPG (MARL)](sota-implementations/multiagent/maddpg_iddpg.py)
    *   [PPO (MARL)](sota-implementations/multiagent/mappo_ippo.py)
    *   [QMIX-VDN (MARL)](sota-implementations/multiagent/qmix_vdn.py)
    *   [SAC (MARL)](sota-implementations/multiagent/sac.py)
    *   [RLHF](examples/rlhf)
    *   [LLM API (GRPO)](sota-implementations/grpo)
*   Code examples
    *   [LLM API & GRPO](sota-implementations/grpo) - Complete language model fine-tuning pipeline
    *   [RLHF](examples/rlhf)
    *   [Memory-mapped replay buffers](examples/torchrl_features)
*   [Tutorials and Demos](https://pytorch.org/rl/stable#tutorials) providing practical use cases.

**Citation**

If using TorchRL, please cite the following:

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

**Installation**

1.  **Create a Virtual Environment:**
    ```bash
    python -m venv torchrl
    source torchrl/bin/activate  # Or: venv\Scripts\activate (Windows)
    ```
    or use conda
    ```bash
    conda create --name torchrl python=3.9
    conda activate torchrl
    ```
2.  **Install PyTorch:** (Follow instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/))
3.  **Install TorchRL:**
    ```bash
    pip3 install torchrl
    ```
    For nightly or local builds, see the detailed installation instructions in the original README.

**Optional Dependencies:** (Install based on your needs)

```bash
pip3 install tqdm tensorboard "hydra-core>=1.1" hydra-submitit-launcher
pip3 install "moviepy<2.0.0"
pip3 install dm_control
pip3 install "gym[atari]" "gym[accept-rom-license]" pygame
pip3 install pytest pyyaml pytest-instafail
pip3 install tensorboard
pip3 install wandb
```

**Support & Contribution**

*   **Issues:** Report bugs in this [repository](https://github.com/pytorch/rl).
*   **Questions:** Ask on the [PyTorch forum](https://discuss.pytorch.org/c/reinforcement-learning/6).
*   **Contributing:**  See the [contribution guide](https://github.com/pytorch/rl/blob/main/CONTRIBUTING.md) for details on forking, submitting issues, and PRs.

**Disclaimer**

This is a PyTorch beta feature. Expect breaking changes, but with a deprecation policy.

**License**

MIT License. See [LICENSE](https://github.com/pytorch/rl/blob/main/LICENSE) for details.