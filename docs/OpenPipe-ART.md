<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents for Real-World Tasks</h1>
</p>

</div>

**ART empowers you to train multi-step agents for complex tasks using GRPO and zero-shot rewards, revolutionizing how you build and improve AI agents.**  [Learn More & Contribute to the Project](https://github.com/OpenPipe/ART)

---

## Key Features

*   **Zero-Shot Rewards with RULER:**  Eliminate the need for manual reward engineering using RULER (Relative Universal LLM-Elicited Rewards), which automatically scores agent trajectories with an LLM-as-judge.
    *   **2-3x faster development:** Significantly reduces time spent on reward function design.
    *   **General-purpose:** Applicable to a wide range of tasks without modifications.
    *   **High Performance:** Achieves performance comparable to or exceeding hand-crafted rewards.
    *   **Easy Integration:** Seamlessly integrates as a drop-in replacement for traditional reward functions.

*   **Open-Source & Flexible Framework:** ART is an open-source RL framework designed for integrating GRPO into any Python application.

*   **Training Loop:**  ART simplifies the agent training process with a modular client-server architecture.
    *   **Client:** Handles agent workflows and communication with the server.
    *   **Server:** Manages inference and training, abstracting away complexity.

*   **Easy Setup:**  Train ART agents from any Python environment.

*   **Modular Design:** Provides convenient wrappers to introduce RL training into **existing applications**.

*   **Train from Anywhere:** Run the ART client on your laptop and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU.

*   **Extensive Integrations:** Integrations with hosted platforms like W&B, Langfuse, and OpenPipe provide flexible observability and **simplify debugging**.

*   **Customizable:** ART is customizable with **intelligent defaults**. You can configure training parameters and inference engine configurations to meet specific needs, or take advantage of the defaults, which have been optimized for training efficiency and stability.

*   **Accelerated Training:** By utilizing the most modern tools, ART allows for rapid fine-tuning and RLHF development of LLMs and other AI models.

---

## Getting Started

### Installation

Install the ART library using pip:

```bash
pip install openpipe-art
```

### Notebooks

Explore pre-built examples to get started quickly:

| Agent Task          | Example Notebook                                                                                                                       | Description                                         |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                |

---

## RULER: Automate Reward Generation

Learn more about the innovative RULER system, which eliminates the need for manual reward engineering:  [RULER Documentation](https://art.openpipe.ai/fundamentals/ruler)

---

## ART•E Agent: Real-World Example

See how ART was used to train a Qwen 2.5 14B agent to beat OpenAI's o3 at email retrieval! [ART•E Agent Blog Post](https://openpipe.ai/blog/art-e-mail-agent)

---

## Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).

---

## Contributing

We welcome contributions!  Please see the [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md) file for details.

---

## Stay Updated

*   [Join our Discord](https://discord.gg/zbBHRUpwf4)
*   [Visit our Documentation](https://art.openpipe.ai)
*   [Read the Blog](https://openpipe.ai/blog)

---

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Acknowledgements

ART is built upon the work of many open-source projects, including: Unsloth, vLLM, trl, torchtune, and SkyPilot.