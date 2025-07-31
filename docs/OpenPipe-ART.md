<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  Train intelligent agents for complex tasks with ease using Agent Reinforcement Trainer (ART).
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Train LLM Agents Effortlessly

ART is an open-source framework designed to simplify the training of multi-step agents for real-world tasks using GRPO (Gradient Reinforcement Policy Optimization).  It provides an ergonomic harness for integrating RL into any Python application, accelerating development and improving agent reliability.  Get started today by exploring the [ART GitHub repository](https://github.com/OpenPipe/ART).

### Key Features

*   **Zero-Shot Reward Engineering with RULER:** Leverage Relative Universal LLM-Elicited Rewards (RULER) to automatically score agent trajectories using an LLM-as-judge.
*   **Accelerated Development:** Reduce development time by 2-3x by eliminating the need for hand-crafted reward functions.
*   **Versatile Task Applicability:** Works across any task without modification, thanks to its general-purpose design.
*   **Strong Performance:** Achieves or surpasses the performance of hand-crafted rewards in various benchmarks.
*   **Easy Integration:** Seamlessly integrate RULER as a drop-in replacement for manual reward functions.
*   **Modular Architecture:**  ART separates the client and server components, simplifying integration into existing applications and allowing for flexible training configurations.
*   **Flexible Training Environment:** Train agents on your local machine, or offload compute to GPU-enabled environments.
*   **Observability and Debugging:** Integrate with platforms like W&B, Langfuse, and OpenPipe for improved insights.

## 📏 RULER:  Revolutionizing Reward Functions

**RULER** (Relative Universal LLM-Elicited Rewards) simplifies reward function engineering by utilizing an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest – **no labeled data, expert feedback, or reward engineering required**.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Example Notebooks

Jumpstart your agent training with these example notebooks:

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)                 | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## Why Choose ART?

*   **Simplified Integration:**  Easily integrate RL into your existing Python applications with ART's convenient wrappers.
*   **Flexible Training Environment:**  Run the ART client locally and utilize an ephemeral, GPU-enabled environment on the ART server.
*   **Enhanced Observability:**  Integrate with platforms like W&B, Langfuse, and OpenPipe for seamless debugging and monitoring.
*   **Intelligent Defaults:**  Leverage optimized default configurations for efficient and stable training, or customize to meet specific needs.

## Installation

Install ART with a simple pip command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Example

Explore how ART can be applied to real-world tasks, such as the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) that was trained using Qwen 2.5 14B to outperform o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART utilizes a client-server architecture with a cyclical training process.

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow (executing multiple rollouts in parallel).
    *   Completion requests are sent to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   A `reward` is assigned to each Trajectory after a rollout is finished.

2.  **Training:**
    *   Grouped Trajectories are sent to the server after each rollout has finished. Inference is blocked during training.
    *   The server trains the model using GRPO, initialized from the latest checkpoint.
    *   The trained LoRA is saved and loaded into vLLM.
    *   Inference is unblocked, and the cycle resumes at step 1.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open a [GitHub issue](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions to ART are welcome!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidance.

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

ART leverages many open-source projects and the wider RL community. Special thanks to:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

And thank you to our partners for their support!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:** The main title is more descriptive, and the first sentence acts as a strong hook, highlighting the core benefit.
*   **Keyword Optimization:**  Keywords like "Agent Reinforcement Trainer," "LLM agents," "RL framework," and "GRPO" are strategically used throughout the document.
*   **Headings and Structure:**  Uses clear headings to organize information, making it easy to scan and understand.  This helps with both SEO and readability.
*   **Bulleted Key Features:**  Highlights the key benefits of using ART in an easy-to-read format.
*   **Concise Descriptions:**  Descriptions are more direct and focused on benefits.
*   **Call to Action:** Encourages users to explore the GitHub repository.
*   **Contextual Links:**  Links are included to relevant resources (documentation, examples, etc.).
*   **SEO-Friendly Formatting:**  Uses markdown formatting to optimize for search engines.
*   **Complete Overview:**  The summary provides a good overview of all the key aspects of the project.
*   **More Comprehensive Summary of Training Loop:** Improved explanation of the client-server interactions in the training loop.
*   **Addresses User Questions:** Provides helpful context to potential users.