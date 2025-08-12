<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  <a href="https://github.com/OpenPipe/ART">
    <img src="https://img.shields.io/github/stars/OpenPipe/ART?style=social" alt="GitHub Stars"/>
  </a>
</p>

</div>

## Supercharge Your LLMs with Agent Reinforcement Trainer (ART)

**ART enables you to train powerful, multi-step agents for complex tasks using Reinforcement Learning from Human Feedback (RLHF) and Grouped Reinforcement Policy Optimization (GRPO).** Build and improve your AI agents with no data, and match or beat SOTA performance. 
[Explore the ART Repository](https://github.com/OpenPipe/ART)

**Key Features:**

*   **Train from Experience:**  ART leverages GRPO to enable LLMs to learn from their interactions and experiences, boosting agent reliability and performance.
*   **Zero-Data Learning (MCP•RL):** Automatically train agents for Model Context Protocol (MCP) servers without needing labeled data by analyzing server tools.
*   **General-Purpose & Flexible:** ART optimizes models for any MCP server and can be adapted to a wide array of agentic tasks.
*   **Easy Integration:** Simple to integrate into your existing projects with a user-friendly client and server architecture.
*   **Accelerated Training:**  Leverages vLLM and other optimized libraries to achieve faster training.
*   **Monitoring and Debugging:** Integrations with platforms such as W&B, Langfuse, and OpenPipe to provide flexible observability.
*   **Strong Performance:** Matches or exceeds SOTA performance in 2/3 benchmarks.
*   **Open Source & Customizable:** ART is fully open-source and offers intelligent defaults, enabling you to customize training parameters and engine configurations.

## Key Use Cases & Examples

ART is designed to handle various agentic tasks. Explore the examples below:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## How ART Works

ART employs a client-server architecture for efficient training:

1.  **Inference:** Your code uses the ART client to perform agentic workflows. Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
2.  **Training:** Trajectories are grouped and sent to the server. The server trains your model using GRPO, saves the LoRA, and loads it into vLLM.

## Installation

Get started with ART by running:

```bash
pip install openpipe-art
```

## Explore ART•E:  A Real-World Example

Discover how ART can be applied in real-world tasks by reviewing the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post where we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 📰 Stay Updated

*   **MCP•RL:** [MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)
*   **AutoRL:** [AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)
*   **RULER:** [RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)
*   **ART·E:** [ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)
*   **ART Trainer:** [ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)
*   [📖 See all blog posts →](https://openpipe.ai/blog)

## 🤝 Contributing

We welcome contributions!  See our [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md) for more information.

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

ART is built on the shoulders of giants, and we are grateful to the open-source community. We especially thank the creators of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)