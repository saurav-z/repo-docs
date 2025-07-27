<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  **Supercharge your LLM agents with ART: a powerful open-source framework for Reinforcement Learning from LLM Feedback.**
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## About ART: Train LLM Agents with Experience

ART is an open-source Reinforcement Learning (RL) framework designed to improve the reliability and performance of LLM agents. It enables LLMs to learn from experience through a streamlined training loop, empowering you to build sophisticated agents for various real-world tasks.  This framework provides an ergonomic harness for integrating GRPO into any python application.

**[Explore the ART Repository](https://github.com/OpenPipe/ART)**

### Key Features of ART:

*   **Easy Integration:** Seamlessly integrates RL training into existing Python applications.
*   **Modular Architecture:** Separates the training server, which abstracts the training complexity.
*   **Flexible Training:** Train agents locally or on cloud GPUs.
*   **Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe for detailed debugging and monitoring.
*   **Customization:** Utilize intelligent defaults for efficiency and stability or configure training parameters as needed.
*   **Supports GRPO**, a state-of-the-art Reinforcement Learning algorithm.
*   **Model Compatibility:** Works with most vLLM/HuggingFace-transformers compatible causal language models.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) drastically simplifies agent training by eliminating the need for hand-crafted reward functions. It uses an LLM-as-judge to automatically score agent trajectories, reducing development time and improving agent performance. Simply define your task in the system prompt, and RULER handles the rest—no labeled data, expert feedback, or reward engineering required.

**Key Benefits of RULER:**

*   **Faster Development:** Accelerate development by skipping reward function engineering.
*   **General-Purpose:** Works across various tasks without modification.
*   **High Performance:** Achieves performance levels that match or exceed hand-crafted rewards in most benchmarks.
*   **Simple Integration:** Easily replace manual reward functions with RULER.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Quick Start: Train Your First Agent

ART offers pre-built notebooks to get you started quickly. Train agents on various tasks, including email retrieval, 2048, and more.

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## 🤖 ART•E Agent

Discover how ART can be applied to real-world problems. Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail how we trained Qwen 2.5 14B to beat o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 ART Training Loop: How it Works

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client interfaces between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration.

**Training Loop Outline:**

1.  **Inference:** Your code uses the ART client to perform an agentic workflow. Completion requests are routed to the ART server.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.

2.  **Training:** Trajectories are grouped and sent to the server. The server trains your model using GRPO. It saves the newly trained LoRA and loads it into vLLM, then resumes inference.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models. If you encounter issues with a specific model, please report it on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🚀 Get Started: Installation

Install the ART library with a simple `pip` command:

```bash
pip install openpipe-art
```

## 🤝 Contributing

ART is actively developed, and contributions are welcome! Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

ART is built on the shoulders of giants. We are grateful to the open-source RL community and, in particular, to the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Special thanks to our partners for helping us test ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7