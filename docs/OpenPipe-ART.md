<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents for Real-World Tasks</h1>
</p>

</div>

**ART empowers you to train sophisticated, multi-step agents using reinforcement learning, revolutionizing how you build AI applications.** ([View on GitHub](https://github.com/OpenPipe/ART))

**Key Features:**

*   🎯 **RULER: Zero-Shot Reward Engineering:** Utilize LLMs to automatically score agent trajectories, eliminating the need for hand-crafted reward functions.
*   🚀 **Fast Development:** Significantly accelerates development time by removing the need for manual reward engineering.
*   ⚙️ **General-Purpose:** Works seamlessly across diverse tasks without requiring modification.
*   📈 **High Performance:** Achieves competitive results, matching or exceeding hand-crafted rewards in numerous benchmarks.
*   🔌 **Easy Integration:** Integrates effortlessly into your existing projects, acting as a drop-in replacement for manual reward functions.
*   💻 **Train from Anywhere:** Run the ART client locally and leverage the ART server for GPU-enabled training environments.
*   🛠️ **Customizable:** Customize training parameters and inference engine configurations to your specific needs.
*   🌐 **Observability & Debugging:** Integrations with platforms like Weights & Biases, Langfuse, and OpenPipe streamline debugging and monitoring.

## Getting Started with ART

ART is an open-source RL framework designed to improve the reliability of LLM agents by enabling them to learn from experience. Integrate GRPO into your Python applications to create agents that can perform multi-step tasks.

### Installation

Install ART effortlessly using pip:

```bash
pip install openpipe-art
```

### Explore with Example Notebooks

Get hands-on with ART using these interactive notebooks:

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

## ART Training Loop: How it Works

ART utilizes a client-server architecture to streamline the RL training process:

1.  **Inference:** Your code leverages the ART client for agentic workflows.
2.  **Completion Requests:**  Routed to the ART server, which runs the model's latest LoRA.
3.  **Trajectory Storage:** Messages are stored as a trajectory.
4.  **Reward Assignment:** Your code assigns a reward after a rollout finishes.
5.  **Trajectory Grouping:** Trajectories are grouped and sent to the server.
6.  **Training Execution:** The server trains your model using GRPO.
7.  **LoRA Saving:** The newly trained LoRA is saved and loaded for the next iteration.

## 💡 Why Choose ART?

*   **Integrate RL Easily:**  ART provides convenient wrappers for incorporating RL training into your existing applications.
*   **Flexible Training:** Train models on your local machine or utilize GPU-enabled environments.
*   **Observability:**  Integrations with tools like Weights & Biases and OpenPipe streamline debugging and monitoring.

## 📚 More Information

*   **ART•E Agent:** Discover how ART trained a Qwen 2.5 14B agent to outperform o3 in email retrieval.  Read the [ART•E Agent blog post](https://openpipe.ai/blog/art-e-mail-agent).
*   **ART News:** Stay updated with the latest research and advancements:
    *   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)**
    *   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)**
    *   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)**
    *   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)**
    *   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)**
    *   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)**
*   **[Documentation](https://art.openpipe.ai)**
*   **[Blog](https://openpipe.ai/blog)**

## 🧑‍🤝‍🧑 Contributing

We welcome contributions! Learn how to contribute in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## 📜 Citation

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

We are grateful to the open-source RL community, particularly the authors of [Unsloth](https://github.com/unslothai/unsloth), [vLLM](https://github.com/vllm-project/vllm), [trl](https://github.com/huggingface/trl), [torchtune](https://github.com/pytorch/torchtune), and [SkyPilot](https://github.com/skypilot-org/skypilot).  Thank you to our partners for helping us test ART!