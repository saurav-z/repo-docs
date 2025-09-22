<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART)</h1>
  <p>
    <b>Supercharge your LLM agents with ART: Train them to excel at real-world tasks using GRPO and advanced reinforcement learning.</b>
  </p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features

ART empowers you to build and train powerful LLM agents with ease:

*   **Zero-Shot Reward Engineering with RULER:** Leverage the power of LLMs to automatically score agent trajectories. No hand-crafted reward functions, labeled data, or expert feedback needed!
*   **Faster Development:** Accelerate your development cycles by eliminating the need for complex reward function engineering.
*   **General Purpose & Easy Integration:** Use ART across any task without modification and seamlessly integrate it into your existing applications.
*   **Open-Source & Flexible:** An open-source RL framework providing a flexible and ergonomic harness for integrating GRPO into any Python application.
*   **Pre-built examples:** Jumpstart your projects with ready-to-use notebooks for various agent tasks.

### ⚖️ RULER: Universal LLM-Elicited Rewards

RULER simplifies reward design using LLMs to automatically score agent behavior, eliminating the need for manual reward function creation.

*   **2-3x Faster Development:** Significantly reduce development time.
*   **General-Purpose:** Works for any task without modification.
*   **Performance:** Matches or exceeds performance of hand-crafted rewards in various benchmarks.
*   **Easy to Integrate:** Simple drop-in replacement for manual reward functions.

```python
# Replace hours of work with a single line of code
# Before:
# def complex_reward_function(trajectory):
#     # 50+ lines of scoring logic...
#     pass

# After:
# judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn More about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART provides a flexible and powerful framework for training LLM agents to excel in real-world tasks. It simplifies the process of integrating GRPO into your applications, enabling you to build more reliable and effective agents. Explore [the docs](https://art.openpipe.ai) to learn more.

## 📒 Example Notebooks

Get started quickly with our example notebooks demonstrating ART's capabilities.

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

## 📰 ART News & Updates

Stay up-to-date on the latest ART developments.

*   🗞️ **[ART LangGraph Integration](https://art.openpipe.ai/integrations/langgraph-integration)** Train LangGraph agents with reinforcement learning.
*   🗞️ **[MCP•RL](https://x.com/corbtt/status/1953171838382817625)**: Train models to effectively use MCP server tools.
*   🗞️ **[AutoRL](https://x.com/mattshumer_/status/1950572449025650733)**: Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** now available for automatic reward generation.
*   🗞️ **[ART•E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)**: Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** for easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Use ART?

ART offers numerous benefits for developing and deploying LLM agents:

*   **Simplified RL Integration:** Convenient wrappers to add RL to your existing applications.
*   **Flexible Training:** Train from your laptop or leverage GPU-enabled environments.
*   **Observability & Debugging:** Integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Intelligent Defaults & Customization:** Optimized defaults with customizable training parameters and inference engine configurations.

## Installation

Install ART with a simple pip command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

See how ART can be used for a real-world task!  Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, which details the training of a Qwen 2.5 14B agent that outperforms o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's functionality is separated into **client** and **server** components.  The ART client is compatible with OpenAI and handles the interface to your code. The server runs independently on a GPU-enabled machine and abstracts away the complexity of the RL loop.

1.  **Inference:**
    1.  Your code uses the ART client to perform an agentic workflow, executing multiple rollouts.
    2.  Completions are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  A `reward` is assigned to the Trajectory when a rollout finishes, based on the LLM's performance.

2.  **Training:**
    1.  Trajectories are grouped and sent to the server after each rollout. Inference is blocked while training.
    2.  The server trains your model using GRPO, starting from the latest checkpoint.
    3.  The server saves the new LoRA and loads it into vLLM.
    4.  Inference resumes, and the loop continues from step 1.

This training loop runs until the specified number of iterations are completed.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 is not currently supported.  Report any model issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART welcomes contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

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

ART is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built upon the work of many in the open source RL community. Special thanks to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg