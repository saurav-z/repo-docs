<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
Train multi-step agents for real-world tasks using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Supercharge Your LLM Agents with RL!

ART is an open-source reinforcement learning (RL) framework designed to effortlessly train powerful, multi-step LLM agents, providing an ergonomic way to integrate GRPO into your Python applications.  [Learn More on GitHub](https://github.com/OpenPipe/ART).

**Key Features & Benefits:**

*   **Effortless Agent Training:** Train multi-step agents for complex tasks with a simplified RL pipeline.
*   **RULER: Zero-Shot Reward Engineering:** Leverage LLMs to automatically score agent trajectories, eliminating the need for manual reward functions.
*   **Faster Development:** Reduce development time by 2-3x by skipping reward function engineering.
*   **General-Purpose Applicability:**  ART works across diverse tasks without modification.
*   **Strong Performance:**  Achieves or surpasses hand-crafted rewards in multiple benchmarks.
*   **Easy Integration:**  Seamlessly integrate ART into existing applications with a drop-in replacement for manual reward functions.
*   **Flexible Training:** Run the ART client locally or on a GPU-enabled server.
*   **Observability & Debugging:** Integrations with platforms like W&B, Langfuse, and OpenPipe streamline debugging.
*   **Customizable & Optimized:** Configure training parameters and inference engines to meet specific needs, with intelligent defaults for efficiency and stability.

## 📏 RULER: Zero-Shot Agent Rewards Explained

**RULER** (Relative Universal LLM-Elicited Rewards) simplifies agent training by using an LLM-as-judge to automatically score agent trajectories. Just define your task in the system prompt, and RULER handles the rest – **no labeled data, expert feedback, or reward engineering required.**

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 🚀 Getting Started with ART: Example Notebooks

Explore how ART trains LLM agents with these interactive examples:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## 🛠️ Installation

Install the ART package in your existing project using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: Real-World Application

Learn about how to train Qwen 2.5 14B to beat o3 at email retrieval by checking out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 ART Training Loop: How it Works

ART uses a client-server architecture for efficient RL training.

1.  **Inference:**
    *   Your code uses the ART client for agentic workflows (e.g. parallel rollouts).
    *   Requests are routed to the ART server, which runs the latest LoRA in vLLM.
    *   Messages are stored in a Trajectory.
    *   A `reward` is assigned to the Trajectory at the end.

2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the new LoRA and loads it into vLLM.
    *   The loop resumes at Step 1.

This continues until the specified iterations are complete.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers causal language models (e.g., those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models)), though it does not support Gemma 3 at this time.  If you have issues with other models, please let us know!

## 🤝 Contribute to ART

ART is actively being developed.  Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

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

This repository is available under the [Apache-2.0 License](LICENSE).

## 🙏 Acknowledgements

ART's development is inspired by the open-source RL community and relies on the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We are thankful for the help of partners in testing ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  "Agent Reinforcement Trainer (ART): Supercharge Your LLM Agents with RL!" grabs attention immediately.
*   **Keyword Optimization:** Repeated use of "Agent Reinforcement Trainer," "LLM agents," "reinforcement learning," and "RL" throughout the README.
*   **Structured Headings:**  Uses proper `<h1>`, `<h2>`, and bolding for easy readability and SEO.
*   **Bulleted Key Features:**  Highlights benefits using bullet points to draw attention.
*   **Clear Explanations:**  Explains RULER and the training loop in simple terms.
*   **Emphasis on Benefits:** Focuses on what users gain (e.g., faster development, easy integration).
*   **Call to Action:** Encourages the user to explore the example notebooks and blog posts.
*   **Internal Linking:** Links to other sections in the README.
*   **External Links:** Uses descriptive anchor text and links to relevant resources (docs, Discord, etc.).
*   **Concise Language:** Removes unnecessary jargon and keeps explanations straightforward.
*   **Model Compatibility:** Specifically mentions model support and how to provide feedback if a model does not work.
*   **SEO Optimization:**  The text is now more likely to appear in search results for related terms.