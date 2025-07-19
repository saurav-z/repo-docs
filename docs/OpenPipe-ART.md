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

## Agent Reinforcement Trainer: Supercharge Your LLM Agents with ART

**ART is an open-source framework that empowers you to train robust and reliable LLM agents for complex tasks, eliminating the need for extensive reward function engineering.** ([Original Repo](https://github.com/OpenPipe/ART))

### Key Features

*   **GRPO Training:** Leverage Gradient-based Reinforcement Policy Optimization for efficient agent learning.
*   **RULER (Relative Universal LLM-Elicited Rewards):** Automatically score agent trajectories using an LLM-as-judge, eliminating the need for hand-crafted reward functions.
*   **Simplified Integration:** Easily integrate ART into your existing Python applications with an ergonomic harness.
*   **Flexible Deployment:** Train agents on your local machine or leverage GPU-enabled environments.
*   **Observability & Debugging:** Integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Customization:** Configure training parameters and inference engines to meet your specific needs.

### 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**

*   **Faster Development:** Reduce development time by 2-3x by skipping reward function engineering entirely.
*   **General-Purpose:** Works across any task without modification.
*   **High Performance:** Matches or exceeds hand-crafted rewards in 3/4 benchmarks.
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

### ART Overview

ART improves agent reliability by allowing LLMs to learn from experience. ART provides an ergonomic harness for integrating GRPO into any python application. For a quick hands-on introduction, run one of the notebooks below. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

### 📒 Notebooks - Train Agents on Real-World Tasks

Explore practical examples with interactive Colab notebooks:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

### Why Use ART?

*   **Easy Integration:** Convenient wrappers for introducing RL training into existing applications.
*   **Flexible Training:** Run the ART client on your laptop, or run on a local GPU.
*   **Enhanced Observability:** Integrations with hosted platforms like W&B, Langfuse, and OpenPipe.
*   **Optimized Defaults:** Customizable with intelligent defaults for training efficiency and stability.

### Installation

Get started quickly by installing the ART package:

```bash
pip install openpipe-art
```

### 🤖 ART•E Agent

Learn from a real-world example!  Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we detail how we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

### 🔁 Training Loop Overview

ART's functionality is divided into a **client** and a **server**. The OpenAI-compatible client is responsible for interfacing between ART and your codebase. Using the client, you can pass messages and get completions from your LLM as it improves. The server runs independently on any machine with a GPU. It abstracts away the complexity of the inference and training portions of the RL loop while allowing for some custom configuration. An outline of the training loop is shown below:

1.  **Inference**

    1.  Your code uses the ART client to perform an agentic workflow (usually executing several rollouts in parallel to gather data faster).
    2.  Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    3.  As the agent executes, each `system`, `user`, and `assistant` message is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to its Trajectory, indicating the performance of the LLM.
2.  **Training**

    1.  When each rollout has finished, Trajectories are grouped and sent to the server. Inference is blocked while training executes.
    2.  The server trains your model using GRPO, initializing from the latest checkpoint (or an empty LoRA on the first iteration).
    3.  The server saves the newly trained LoRA to a local directory and loads it into vLLM.
    4.  Inference is unblocked and the loop resumes at step 1.

This training loop runs until a specified number of inference and training iterations have completed.

### 🧩 Supported Models

ART should work with most vLLM/HuggingFace-transformers compatible causal language models, or at least the ones supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Gemma 3 does not appear to be supported for the time being. If any other model isn't working for you, please let us know on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

### 🤝 Contributing

ART welcomes contributions! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### 📖 Citation

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

### ⚖️ License

This project is available under the [Apache-2.0 License](LICENSE).

### 🙏 Credits

ART relies on the work of many. We're especially grateful to the authors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who've helped us test ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7
```
Key improvements and optimizations:

*   **SEO-Friendly Title & Introduction:**  Uses the target keyword "Agent Reinforcement Trainer" and a clear, concise opening sentence.
*   **Clear Headings:**  Uses headings for better readability and organization, crucial for SEO.
*   **Keyword Density:** Includes the primary keyword "Agent Reinforcement Trainer" and related terms throughout.
*   **Bulleted Key Features:**  Emphasizes key benefits, making them easy to scan.
*   **Action-Oriented Language:** Uses strong verbs ("Supercharge," "Empowers," "Train") to engage the reader.
*   **Call to Action:** Encourages exploration with links to the documentation and examples.
*   **Concise Summaries:** Streamlines the text while retaining essential information.
*   **Proper Formatting:**  Uses Markdown for headings, lists, and code blocks.
*   **Links:** Includes links to the original repo, docs, and other relevant resources.
*   **Targeted Keywords:** Incorporated keywords like "LLM agents," "RL training," "GRPO," and "RULER"
*   **Removed Redundancy:**  Condensed some repetitive information.
*   **Readability:** Improved the overall flow and readability of the content.