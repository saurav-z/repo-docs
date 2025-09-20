<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

</div>

# Agent Reinforcement Trainer (ART): Train LLM Agents with Ease

**Revolutionize your LLM agent development with Agent Reinforcement Trainer (ART), an open-source framework that empowers you to build and refine multi-step agents for real-world tasks.**

[View the original repository on GitHub](https://github.com/OpenPipe/ART)

**Key Features:**

*   **Zero-Shot Reward Engineering with RULER:**  Use LLMs to automatically score agent performance, eliminating the need for complex reward functions.
*   **Accelerated Development:** Reduce development time by 2-3x by skipping reward function engineering.
*   **General-Purpose Applicability:** Works seamlessly across a variety of tasks without modification.
*   **Performance:** Achieve strong performance, often matching or exceeding hand-crafted rewards.
*   **Easy Integration:**  Drop-in replacement for manual reward functions.
*   **Modular Architecture:**  ART provides an ergonomic harness for integrating GRPO into any Python application, with a client-server architecture.
*   **Flexible Training:** Train from anywhere - laptop or cloud - with integrations for observability (Weights & Biases, Langfuse, OpenPipe).
*   **Customizable and Intelligent Defaults:** Configure training parameters and inference settings to meet specific needs or leverage optimized defaults.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

## 🚀 Why Use ART?

ART simplifies the process of training agents by leveraging GRPO (Generative Reinforcement Policy Optimization) and RULER (Relative Universal LLM-Elicited Rewards). It offers:

*   **Simplified RL Training:** Train LLM-based agents more easily with a streamlined GRPO implementation.
*   **Rapid Prototyping:**  Accelerate your development cycle with zero-shot reward capabilities.
*   **Improved Agent Reliability:** Enhance your agents' performance through iterative learning.
*   **Integration with Existing Applications:** Convenient wrappers make it easy to introduce RL training into your current projects.

## 📒 Example Notebooks

Get started quickly with these examples showcasing ART's capabilities:

| Agent Task          | Example Notebook                                                                                                                       | Description                                         | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server              | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue           | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe              | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/raw/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames                | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                | [Link coming soon]                                                                                                                                                                                                          |

## 📰 ART News

Stay updated on the latest advancements and integrations:

*   **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## ⚙️ Installation

Install ART using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent Example

Discover how ART can be applied to real-world tasks with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, showcasing how Qwen 2.5 14B was trained to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART employs a client-server architecture:

1.  **Inference (Client):** Your code utilizes the ART client to execute agentic workflows. Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
2.  **Training (Server):** Trajectories are grouped and sent to the server after rollouts.  The server trains your model using GRPO, saves the updated LoRA, and loads it into vLLM.

This loop continues until the specified number of iterations is complete.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models. If you encounter any issues with a specific model, please report it on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md) for details.

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

ART leverages the work of the open-source RL community.  Special thanks to the projects:  [Unsloth](https://github.com/unslothai/unsloth), [vLLM](https://github.com/vllm-project/vllm), [trl](https://github.com/huggingface/trl), [torchtune](https://github.com/pytorch/torchtune), and [SkyPilot](https://github.com/skypilot-org/skypilot).