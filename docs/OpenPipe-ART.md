<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART)</h1>
</div>

**Supercharge your LLM agents with ART, the open-source framework for training multi-step agents for real-world tasks using GRPO and zero-shot rewards.**  ([Original Repo](https://github.com/OpenPipe/ART))

**Key Features:**

*   **Zero-Shot Reward Engineering with RULER:** Eliminate the need for hand-crafted reward functions. RULER (Relative Universal LLM-Elicited Rewards) automatically scores agent trajectories using an LLM-as-judge, saving development time and effort.
*   **Rapid Development:**  Accelerate your development cycles with ART, reducing the time spent on reward function engineering.
*   **General Purpose:** ART works seamlessly across various tasks without any modifications.
*   **Proven Performance:**  ART achieves strong performance, often matching or surpassing hand-crafted rewards.
*   **Easy Integration:** Integrate ART into your existing applications with a simple, drop-in replacement for manual reward functions.
*   **Modular Architecture:** ART's architecture divides into a client and server, allowing training from anywhere.
*   **Flexible Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe offer enhanced observability.
*   **Customizable Training:** Configurable training parameters and inference engine configurations to meet specific needs.
*   **Wide Model Support:**  Works with most vLLM/HuggingFace-transformers compatible causal language models.

## Table of Contents

*   [RULER: Zero-Shot Agent Rewards](#ruler-zero-shot-agent-rewards)
*   [ART Overview](#art-overview)
*   [Notebooks](#notebooks)
*   [ART News](#art-news)
*   [Why ART?](#why-art)
*   [Installation](#installation)
*   [ART•E Agent](#art-e-agent)
*   [Training Loop Overview](#training-loop-overview)
*   [Supported Models](#supported-models)
*   [Contributing](#contributing)
*   [Citation](#citation)
*   [License](#license)
*   [Credits](#credits)

## RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by using an LLM-as-judge to automatically score agent trajectories. By simply defining your task in the system prompt, RULER handles the rest. **No labeled data, expert feedback, or reward engineering is required.**

✨ **Key Benefits:**

*   **2-3x faster development** - Skip reward function engineering entirely
*   **General-purpose** - Works across any task without modification
*   **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
*   **Easy integration** - Drop-in replacement for manual reward functions

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART is an open-source RL framework designed to enhance agent reliability by leveraging LLMs to **learn from experience**.  ART provides an ergonomic harness for integrating GRPO into any python application.

## Notebooks

Explore these example notebooks to get hands-on with ART:

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

## ART News

Stay up-to-date with the latest developments in ART:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why ART?

*   **Simplified Integration:**  ART makes it easy to introduce RL training into your existing applications with convenient wrappers.
*   **Flexible Training:** Train from your laptop, on a local GPU, or kick off an ephemeral GPU-enabled environment.
*   **Enhanced Observability:** Integrations with W&B, Langfuse, and OpenPipe simplify debugging.
*   **Optimized Defaults:** ART comes with intelligent defaults, allowing for easy configuration.

## Installation

Get started by installing ART:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn how ART is used for real-world tasks by reading the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where the Qwen 2.5 14B email agent beat o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART functions as a **client** and a **server**. The OpenAI-compatible client interfaces with your code. The server handles the inference and training.

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

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you encounter any compatibility issues, please report them on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

Contributions to ART are welcome! See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

This project is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART builds upon the work of many. Special thanks to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

And thank you to our partners for helping us test ART!