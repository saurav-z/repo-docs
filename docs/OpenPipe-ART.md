<div align="center">
    <a href="https://art.openpipe.ai">
        <picture>
            <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
        </picture>
    </a>
    <p align="center">
        <h1>Agent Reinforcement Trainer (ART)</h1>
    </p>
</div>

**Revolutionize multi-step agent training with ART, an open-source framework that empowers LLMs to learn from experience.** ([Back to Original Repo](https://github.com/OpenPipe/ART))

## Key Features

*   **Zero-Shot Reward Engineering with RULER:** Automate reward scoring using an LLM-as-judge.
*   **Faster Development:** Reduce development time by eliminating the need for manual reward functions.
*   **General Purpose:** Works across diverse tasks without modification.
*   **High Performance:** Achieve or surpass the performance of hand-crafted reward functions.
*   **Easy Integration:** Seamlessly integrate with existing applications as a drop-in replacement for reward functions.
*   **Modular Architecture:**  ART is divided into a client and server for flexibility and ease of use.
*   **Flexible Training:** Train agents on your local machine or leverage cloud resources, including GPU-enabled environments.
*   **Observability Integrations:** Leverage integrations with platforms like W&B, Langfuse, and OpenPipe for streamlined debugging and monitoring.

## What is ART?

ART is an open-source Reinforcement Learning (RL) framework designed to improve the reliability of agents by enabling LLMs to learn from their experiences. This is done with Generalized Reinforcement Policy Optimization (GRPO), allowing agents to tackle real-world tasks with unprecedented efficiency and performance. ART provides an ergonomic harness for integrating GRPO into any python application.

### Why Choose ART?

*   **Simplified Integration:** Easily add RL training to your existing applications.
*   **Flexible Training:** Train agents locally or in the cloud, with options for GPU acceleration.
*   **Intelligent Defaults:** Benefit from optimized default configurations for training efficiency.

## 📏 RULER: Zero-Shot Agent Rewards

RULER (Relative Universal LLM-Elicited Rewards) simplifies agent training by eliminating the need for hand-crafted reward functions. It uses an LLM-as-judge to automatically score agent trajectories, simplifying the process and saving development time.

**Key Advantages:**

*   **Faster Development (2-3x speedup):** Eliminates manual reward engineering.
*   **Universal Applicability:** Works across diverse tasks without code changes.
*   **Competitive Performance:** Matches or exceeds hand-crafted rewards in most benchmarks.
*   **Easy to Integrate:** Drop-in replacement for manual reward functions.

```python
# Before: Laborious Reward Engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: Simple with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Example Notebooks

Explore various use cases with these example notebooks:

| Agent Task          | Example Notebook                                                                                                                       | Description                                                | Comparative Performance                                                                                                                                                                                                     |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E LangGraph** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/langgraph/art-e-langgraph.ipynb)   | Qwen 2.5 7B learns to search emails using LangGraph         | [Link coming soon]                                                                                                                                                                                                          |
| **MCP•RL**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server                     | [Link coming soon]                                                                                                                                                                                                          |
| **ART•E [RULER]**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/dev/art-e/art_e/evaluate/display_benchmarks.ipynb)                              |
| **2048**            | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/display_benchmarks.ipynb)                                                |
| **Temporal Clue**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue                   | [Link coming soon]                                                                                                                                                                                                          |
| **Tic Tac Toe**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/display-benchmarks.ipynb)                            |
| **Codenames**       | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)                | Qwen 2.5 3B learns to play Codenames                     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](https://github.com/OpenPipe/art-notebooks/blob/main/examples/codenames/Codenames_RL.ipynb) |
| **AutoRL [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task                      | [Link coming soon]                                                                                                                                                                                                          |

## 📰 ART News & Updates

Stay informed about the latest advancements in ART.

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[See All Blog Posts](https://openpipe.ai/blog)

## 🤖 ART•E Agent

Learn how ART can be used to create effective agents by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post. See how Qwen 2.5 14B was trained to outperform o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's functionality is structured into a **client** and a **server**, where the client provides the interface with ART, and the server handles training and inference.

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training:**
    *   When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    *   The server trains your model using GRPO.
    *   The server saves the newly trained LoRA and loads it into vLLM.
    *   Inference is unblocked and the loop resumes at step 1.

The training loop runs until a specified number of iterations is complete.

## Installation

Get started by installing ART:

```bash
pip install openpipe-art
```

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models.

*   **Note:** Gemma 3 is not currently supported.

For any model compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details.

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

ART's development would not have been possible without the contributions of the open source RL community, especially the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)