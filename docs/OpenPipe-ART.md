<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART): Train LLM Agents with Ease</h1>
</p>

</div>

**ART empowers you to build and train advanced AI agents for real-world tasks using reinforcement learning, allowing your LLMs to learn from experience.**  ([View on GitHub](https://github.com/OpenPipe/ART))

## Key Features

*   **Zero-Shot Reward Engineering:**  Leverage **RULER** (Relative Universal LLM-Elicited Rewards) to automatically score agent trajectories using an LLM-as-judge, eliminating the need for hand-crafted reward functions.
*   **Simplified Development:** Significantly accelerates the development process by eliminating the need for complex reward function engineering.
*   **General-Purpose Applicability:** Works seamlessly across various tasks without requiring task-specific modifications.
*   **Enhanced Performance:**  Achieves performance comparable to or surpassing hand-crafted rewards in numerous benchmarks.
*   **Easy Integration:**  Provides a simple drop-in replacement for manual reward functions.
*   **Open-Source & Customizable:** An open-source RL framework built with intelligent defaults. Customize training parameters and inference engine configurations to meet specific needs.
*   **Flexible Training:**  Train agents from any client machine with Python. Easily run the ART client and let the ART server deploy to an ephemeral GPU-enabled environment, or run on a local GPU.

## RULER: Zero-Shot Agent Rewards - The Future of Reward Engineering

**RULER** revolutionizes agent training by removing the need for manual reward functions.  Define your task with a system prompt, and RULER does the rest, **requiring no labeled data, expert feedback, or reward engineering**.

**Key Benefits of RULER:**

*   **Faster Development:** 2-3x faster development by skipping reward function engineering.
*   **Universal Applicability:** Works seamlessly across any task without modification.
*   **Strong Performance:** Matches or exceeds hand-crafted rewards in 3/4 benchmarks.
*   **Easy Integration:** Drop-in replacement for manual reward functions.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART: An Overview

ART is an open-source Reinforcement Learning (RL) framework that enhances agent reliability by allowing Large Language Models (LLMs) to **learn from experience**. ART provides an ergonomic harness for integrating GRPO into any Python application.  Check out the [docs](https://art.openpipe.ai) to learn more.

## 📒 Example Notebooks: Train Agents on Diverse Tasks

Explore our notebooks to get hands-on experience training agents across a variety of tasks.

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

## 📰 Stay Updated with ART News

Stay informed about our latest research, integrations, and updates on building state-of-the-art AI agents.

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** Convenient wrappers for integrating RL training into existing applications. ART abstracts the training server into a modular service.
*   **Flexible Training Environment:** Train from anywhere! Run the ART client on your laptop, and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU.
*   **Enhanced Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe provide flexible observability and simplify debugging.
*   **Intelligent Defaults & Customization:** ART is customizable with intelligent defaults. Configure training parameters and inference engine configurations to meet specific needs.

## Installation

Get started by installing the `openpipe-art` package:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Explore the capabilities of ART by checking out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, which details the training of a Qwen 2.5 14B agent that outperforms o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Understanding the Training Loop

ART operates with a client-server architecture. The client interacts with your code, and the server handles the training.

1.  **Inference:**
    1.  Your code uses the ART client to perform an agentic workflow.
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory.

2.  **Training:**
    1.  When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    3.  The server saves the newly trained LoRA to a local directory and loads it into vLLM.
    4.  Inference is unblocked and the loop resumes at step 1.

This training loop continues until a specified number of iterations are complete.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, as supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).

## 🤝 Contribute

ART is under active development. We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

This repository's source code is available under the [Apache-2.0 License](LICENSE).

## 🙏 Acknowledgements

ART leverages the work of many open-source projects and individuals. We're particularly grateful to the contributors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

A special thanks to our partners who helped us test ART!