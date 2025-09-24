<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

## 🚀 Train LLM Agents with Ease using ART

**ART (Agent Reinforcement Trainer)** is an open-source framework that simplifies training multi-step agents for complex, real-world tasks using GRPO. It empowers developers to build and refine intelligent agents, **accelerating development and improving agent reliability.**  Dive deeper into the capabilities of ART at the [original repository](https://github.com/OpenPipe/ART).

**Key Features:**

*   **Simplified Training:** Train agents with GRPO, abstracting away the complexities of the training server.
*   **Zero-Shot Rewards with RULER:** Leverage LLMs as judges with RULER to automatically score agent trajectories, eliminating the need for manual reward function engineering.
*   **Faster Development:** Speed up development by eliminating reward function engineering.
*   **Versatile:** ART works across diverse tasks without modification.
*   **Strong Performance:** Achieve competitive results with hand-crafted rewards.
*   **Easy Integration:** Seamlessly integrates into your existing applications.
*   **Flexible Training:** Train from any machine with Python and run the ART client on your local machine, or leverage a GPU-enabled environment.
*   **Observability & Debugging:** Integrated with platforms like W&B, Langfuse, and OpenPipe.
*   **Customizable & Optimized Defaults:** Configure training parameters or use optimized defaults.

## 📏 RULER: Revolutionizing Agent Rewards

**RULER (Relative Universal LLM-Elicited Rewards)** simplifies reinforcement learning by using an LLM-as-judge to automatically score agent trajectories. This innovative approach eliminates the need for hand-crafted reward functions, enabling faster development cycles.

*   **2-3x Faster Development:** Skip the tedious reward function engineering.
*   **General-Purpose:** Applicable across various tasks without modifications.
*   **High Performance:** Achieve results that match or surpass hand-crafted rewards.
*   **Easy Integration:** Streamline reward functions with a simple drop-in replacement.

```python
# Before: Laborious reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of scoring logic...
    pass

# After: Concise with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[Learn more about RULER](https://art.openpipe.ai/fundamentals/ruler)

## 📒 ART Example Notebooks

Experiment with ART through these hands-on notebooks:

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

Stay informed with the latest advancements in ART:

*   **ART integrates with LangGraph**: Train your LangGraph agents with reinforcement learning.
*   **MCP•RL: Teach Your Model to Master Any MCP Server**: Automatically train models to effectively use MCP server tools through reinforcement learning.
*   **AutoRL: Zero-Data Training for Any Task**: Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   **RULER: Easy Mode for RL Rewards**: RULER is now available for automatic reward generation in reinforcement learning.
*   **ART·E: How We Built an Email Research Agent That Beats o3**:  Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   **ART Trainer: A New RL Trainer for Agents**: Enables easy training of LLM-based agents using GRPO.

[See all blog posts](https://openpipe.ai/blog)

## ⚙️ How ART Works

ART is built around a client-server architecture to facilitate agent training.  The client interfaces with your code, while the server handles the complexities of training.

### 🔁 Training Loop Overview

1.  **Inference:**
    1.  Your code uses the ART client to perform agentic workflows (e.g., multiple parallel rollouts).
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training:**
    1.  Trajectories are grouped and sent to the server. Inference is blocked during training.
    2.  The server trains your model using GRPO, starting from the latest checkpoint.
    3.  The server saves the newly trained LoRA and loads it into vLLM.
    4.  Inference resumes.

This loop continues for a specified number of iterations.

## 📦 Installation

Install ART to integrate it into your project:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent Example

Explore how ART can be used in real-world applications by checking out the [ART•E Agent blog post](https://openpipe.ai/blog/art-e-mail-agent), where they trained Qwen 2.5 14B to excel at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models. Check the [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) documentation for the latest compatibility.  If you experience any model-related issues, please contact the project on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

Contributions to ART are welcome! Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART builds upon the work of several projects. We are especially grateful to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART!