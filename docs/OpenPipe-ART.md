<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
Train powerful, multi-step agents for real-world tasks efficiently using GRPO.
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![Downloads][downloads-image]][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Supercharge Your LLM Agents 🚀

**ART is an open-source framework that empowers you to train and refine LLM-powered agents for complex tasks, enabling them to learn from experience.**  This allows for the creation of powerful and reliable agents.  [Check out the original repo](https://github.com/OpenPipe/ART).

**Key Features:**

*   **GRPO-Based Training:** Leverage the power of GRPO (Gradient Reinforcement Policy Optimization) for efficient agent training.
*   **RULER: Zero-Shot Reward Function:** Implement automatic scoring of agent trajectories using an LLM-as-judge. No hand-crafted reward functions, labeled data, or expert feedback required.
*   **Easy Integration:**  ART provides Python wrappers for seamless integration into existing applications.
*   **Flexible Deployment:** Train agents locally or in the cloud with ephemeral GPU-enabled environments.
*   **Observability:** Integrate with platforms like W&B, Langfuse, and OpenPipe for enhanced debugging and monitoring.
*   **Customizable:** Tailor training parameters and inference configurations to meet your specific needs.
*   **Pre-built examples:** Ready-to-use notebooks to train on various tasks.
*   **Modular Client/Server Architecture:** The ART client and server architecture abstracts the training loop's complexities.

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**

*   **Faster Development:** Reduce development time by 2-3x by skipping reward function engineering entirely.
*   **Universal Applicability:** Works across any task without modification.
*   **High Performance:** Matches or exceeds hand-crafted rewards in 3/4 benchmarks.
*   **Simple Integration:** Drop-in replacement for manual reward functions.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## 📒 Example Notebooks: Get Started Quickly

ART offers several example notebooks to kickstart your agent training:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Use ART?

*   **Simplified RL Integration:** Easily incorporate RL training into existing applications with ART's wrappers.
*   **Flexible Training Environments:** Run the ART client on your local machine and leverage a remote GPU server for training, or train on a local GPU.
*   **Enhanced Debugging & Monitoring:** Integrate with platforms like W&B, Langfuse, and OpenPipe.
*   **Optimized Defaults:** Leverage ART's intelligent default settings to train efficiently and reliably.

## 🚀 Installation

Install the ART library using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent Example

Learn about a real-world application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post!  See how Qwen 2.5 14B was trained to beat o3 at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Breakdown

ART's functionality is divided into a **client** and a **server**.

1.  **Inference:**
    *   Your code uses the ART client to perform an agentic workflow.
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each system, user, and assistant message is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a reward to its Trajectory.

2.  **Training:**
    *   Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    *   The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    *   The server saves the trained LoRA and loads it into vLLM.
    *   Inference is unblocked, and the loop repeats.

This training loop runs until the defined number of iterations is complete.

## 🧩 Supported Models

ART is designed to work with a wide variety of vLLM/HuggingFace-transformers compatible causal language models.  Refer to [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) to see the supported models.  Contact us via [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues) for support.

## 🤝 Contributing

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

ART builds upon the work of many, with special thanks to the creators of:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who have helped with testing!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and explanations:

*   **SEO-Optimized Title & Introduction:** Added a concise hook and descriptive title including keywords like "Agent Reinforcement Trainer," and "LLM Agents".
*   **Clear Headings:**  Organized content with clear, descriptive headings.
*   **Bulleted Key Features:** Made the benefits and features easy to scan.
*   **Concise Language:**  Streamlined the language for better readability and clarity.
*   **Stronger Calls to Action:**  Encourages engagement with examples and links.
*   **Example-Driven:** Emphasizes examples and practical applications.
*   **Emphasis on Benefits:** Highlights *why* someone would use ART.
*   **Installation and Usage Focused:**  Provides clear, actionable instructions.
*   **Added keyword rich intro sentence to the Agent Reinforcement Trainer section**
*   **Simplified the training loop overview.**
*   **Expanded on Key Features to be more descriptive**
*   **Revised the "Why ART?" section for clarity and conciseness**
*   **Included Alt text for image tags**
*   **Improved the formatting and markdown syntax**