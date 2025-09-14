<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents with Ease</h1>
</div>

**ART empowers you to train multi-step agents for real-world tasks using cutting-edge Reinforcement Learning (RL) techniques.** [Learn more at the original repo](https://github.com/OpenPipe/ART).

**Key Features:**

*   ✨ **RULER: Zero-Shot Rewards:** Automatically score agent trajectories using an LLM-as-judge, eliminating the need for hand-crafted reward functions.
*   🚀 **Accelerated Development:**  Reduce development time by 2-3x by skipping reward engineering.
*   🎯 **General Purpose:** Works across various tasks without modification.
*   📈 **Performance:** Achieves strong performance, matching or exceeding hand-crafted rewards in many benchmarks.
*   🔌 **Easy Integration:** Seamlessly integrates as a drop-in replacement for manual reward functions.
*   💻 **Flexible Training:** Train from anywhere, from your laptop to a GPU-enabled environment.
*   🧩 **Modular Architecture:** Convenient wrappers for introducing RL training into existing applications.
*   📊 **Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe simplify debugging.
*   ⚙️ **Customization:**  Configure training parameters and inference engine configurations with intelligent defaults.

## 🧠 What is ART?

ART is an open-source RL framework built for improving the reliability of agents by allowing LLMs to **learn from experience.** It offers a streamlined environment for integrating GRPO (Gradient-Based Reinforcement Policy Optimization) into any Python application, allowing you to quickly train and deploy sophisticated AI agents.

### Quick Start:

*   **Installation:** `pip install openpipe-art`
*   **Explore Examples:** Jumpstart your training with our interactive notebooks:

## 📒 Example Notebooks
| Agent Task | Notebook | Description | Performance |
|---|---|---|---|
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon] |
| **MCP•RL** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb) | Qwen 2.5 3B masters the NWS MCP server | [Link coming soon] |
| **ART•E [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb) | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb) | Qwen 2.5 3B learns to play 2048 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb) |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon] |
| **Tic Tac Toe** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb) | Qwen 2.5 3B learns to play Tic Tac Toe | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb) |
| **Codenames** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) | Qwen 2.5 3B learns to play Codenames | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb) | Train Qwen 2.5 7B to master any task | [Link coming soon] |

## 📰 ART News

*   **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## 🤖 ART•E Agent

Explore how to use ART in a real-world scenario with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, demonstrating how a Qwen 2.5 14B agent was trained to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Simplified

ART utilizes a client-server architecture for efficient training. Here's a breakdown:

1.  **Inference (Client):**  Your code interacts with the ART client to perform agentic workflows.
2.  **Completion Requests (Server):** The ART server runs the latest LoRA of the LLM using vLLM.
3.  **Trajectory Storage:**  `system`, `user`, and `assistant` messages are stored as a `Trajectory`.
4.  **Reward Assignment:**  After a rollout, your code provides a `reward`.
5.  **Training (Server):** Trajectories are grouped and sent to the server for GRPO training using the latest checkpoint.
6.  **LoRA Update:** The server saves the trained LoRA and loads it into vLLM.
7.  **Loop Resumes:**  Inference is unblocked, and the cycle continues.

This iterative loop continues until a defined number of iterations are reached.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers-compatible causal language models, specifically those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). If you encounter any model compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or create an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contribute

We encourage community contributions!  Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is released under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built on the work of many contributors. We are especially grateful to the authors of:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

We are thankful to our partners for helping us test ART in the wild! We look forward to what you build!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg