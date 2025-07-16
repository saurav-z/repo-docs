<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
Train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Train Powerful LLM Agents with Ease using ART

ART (Agent Reinforcement Trainer) is an open-source framework designed to simplify the training of multi-step LLM agents for real-world tasks, leveraging the power of GRPO (Guided Policy Optimization).  **Accelerate your agent development and build reliable, high-performing AI agents with ART!**  [Explore the ART repository](https://github.com/OpenPipe/ART).

**Key Features:**

*   **RULER for Zero-Shot Reward Engineering:**  Use Relative Universal LLM-Elicited Rewards (RULER) to score agent trajectories automatically, eliminating the need for complex reward functions.
*   **Simplified Training Loop:** ART provides an ergonomic framework for integrating GRPO into any Python application, abstracting the training server for ease of use.
*   **Flexible Deployment:** Train agents from anywhere, locally or with GPU-enabled environments, and integrate with platforms like W&B, Langfuse, and OpenPipe for enhanced debugging and observability.
*   **Customizable with Intelligent Defaults:** Configure training parameters or use optimized defaults for efficient and stable training.
*   **Ready-to-Use Notebooks:** Get started quickly with example notebooks for various agent tasks like email searching, 2048, and Tic Tac Toe.
*   **Open Source and Community Driven:**  Contribute to the development of ART and build with the community.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**
- **2-3x faster development** - Skip reward function engineering entirely
- **General-purpose** - Works across any task without modification
- **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
- **Easy integration** - Drop-in replacement for manual reward functions

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Example Notebooks - Train Your First Agent

ART provides example notebooks to get you started quickly with training agents for various tasks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Use ART?

*   **Easy Integration**: Seamlessly integrate RL into your existing applications.
*   **Flexible Training**: Train agents locally or leverage GPU-enabled environments.
*   **Enhanced Observability**: Integrate with platforms like W&B and Langfuse for debugging.
*   **Optimized Defaults**: Benefit from intelligent defaults optimized for training efficiency and stability.

## Installation

Install ART with pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to see how Qwen 2.5 14B was trained to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1.  **Inference**

    1.  Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.
2.  **Training**

    1.  When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    3.  The server saves the newly trained LoRA to a local directory and loads it into vLLM.
    4.  Inference is unblocked and the loop resumes at step 1.

This training loop runs until a specified number of inference and training iterations have completed.

## 🧩 Supported Models

ART should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being. If any other model isn't working for you, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## 📖 Citation

```bibtex
@misc{hilton2025art,
  author = {Brad Hilton and Kyle Corbitt and David Corbitt and Saumya Gandhi and Angky William and Bohdan Kovalenskyi and Andie Jones},
  title = {ART: Agent Reinforcement Trainer},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/openpipe/art}}
}
```

## ⚖️ License

This repository is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

We're grateful to the open source RL community and the projects below:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thanks to our partners for helping test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7