<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

## 🚀 Train Smarter AI Agents with ART: The Open-Source Reinforcement Learning Framework

**ART (Agent Reinforcement Trainer)** empowers you to train powerful, multi-step AI agents for real-world tasks, eliminating the need for extensive reward engineering and accelerating development. Explore the original repository on [GitHub](https://github.com/OpenPipe/ART).

**Key Features:**

*   **Zero-Shot Reward Engineering with RULER:** Leverage the power of LLMs to automatically score agent trajectories. Simply define your task, and RULER handles the rest.
*   **Faster Development:** Significantly reduce development time by eliminating the need for manual reward function creation.
*   **General-Purpose Applicability:** Train agents across any task without requiring task-specific modifications.
*   **High Performance:** Achieve performance that matches or exceeds hand-crafted rewards in many benchmarks.
*   **Easy Integration:** Seamlessly integrate ART into your existing Python applications.
*   **Open Source & Customizable:**  Benefit from a flexible, open-source framework with intelligent defaults that are easy to customize to your needs.

## 🧠 RULER: Revolutionizing Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards)** drastically simplifies the agent training process.  Instead of spending hours crafting complex reward functions, RULER uses an LLM-as-judge to automatically assess agent performance, freeing you to focus on agent design and evaluation.

**Benefits of Using RULER:**

*   **2-3x Faster Development:** Skip reward function engineering entirely.
*   **Versatile:** Works across various tasks without modification.
*   **Effective Performance:** Achieves strong results, often comparable to or better than hand-crafted reward functions.
*   **Simplified Implementation:** Easy drop-in replacement for manual reward functions.

```python
# Before: Complex, time-consuming reward function
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: Effortless evaluation with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER](https://art.openpipe.ai/fundamentals/ruler)

## 🛠️ ART Overview: Learn from Experience

ART is an open-source RL framework designed to enhance agent reliability by enabling LLMs to learn from experience. It provides an ergonomic harness for integrating GRPO (Gradient-based Reinforcement Policy Optimization) into any Python application.

## 📒 Example Notebooks: Get Started Quickly

Jumpstart your agent training with these hands-on notebooks:

| Agent Task          | Example Notebook                                                                                                                       | Description                                                                                                                                                            | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph                                                                                                          | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server                                                                                                                        | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E \[RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER                                                                                                                  | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                                                                                                                                    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue                                                                                                                          | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe                                                                                                                             | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                                                                                                                              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL \[RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                                                                                                                              | [Link coming soon]                                                                                                                                                                                                          |

## 📰 Stay Updated: ART News & Research

*   🗞️ **[ART Integrates with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)**: Train your LangGraph agents for enhanced multi-step reasoning and tool usage.
*   🗞️ **[MCP•RL: Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)**: Train models to effectively use MCP server tools using reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training](https://x.com/mattshumer_/status/1950572449025650733)**: Train custom AI models without labeled data, leveraging automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)**: Automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: Email Agent](https://openpipe.ai/blog/art-e-mail-agent)**: Learn how an ART agent outperformed o3 in email retrieval.
*   🗞️ **[ART Trainer](https://openpipe.ai/blog/art-trainer)**: A new RL Trainer for Agents that enables easy training of LLM-based agents using GRPO.

[📖 Explore all blog posts](https://openpipe.ai/blog)

## ✨ Why Choose ART?

*   **Seamless Integration:** ART provides convenient wrappers for incorporating RL training into existing applications.
*   **Flexible Training:** Train agents on your laptop or in a GPU-enabled environment, or on a local GPU.
*   **Observability & Debugging:** Integrate with platforms like W&B, Langfuse, and OpenPipe for streamlined debugging.
*   **Intelligent Defaults:** Benefit from optimized training parameters and inference engine configurations.

## ⚙️ Installation

Install ART easily using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Discover how ART was used to train an email retrieval agent that surpassed o3 in performance. Explore the details in the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop: How ART Works

ART operates with a client-server architecture to manage the RL training process:

1.  **Inference:**
    1.  Your code uses the ART client to perform an agentic workflow.
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each message is stored in a Trajectory.
    4.  A `reward` is assigned to each Trajectory.

2.  **Training:**
    1.  Trajectories are grouped and sent to the server.
    2.  The server trains the model using GRPO.
    3.  The server saves the trained LoRA.
    4.  The loop repeats.

This loop continues until a specified number of iterations are complete.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models.  If you experience any compatibility issues, please report them on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contribute to ART

We welcome contributions! Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

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

ART is built upon the contributions of the open-source RL community. We're particularly grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We appreciate our partners for their support. We're excited to see what you create with ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg