<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART)</h1>

  <p><b>Supercharge your LLM agents: ART is an open-source framework that lets LLMs learn from experience to perform complex, real-world tasks.</b></p>

  [![PRs-Welcome][contribute-image]][contribute-url]
  [![Downloads][downloads-image]][pypi-url]
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features

*   **Train Multi-Step Agents:** Build and refine LLM agents capable of handling complex, multi-step tasks using Reinforcement Learning (RL).
*   **Zero-Shot Reward with RULER:** Leverage RULER (Relative Universal LLM-Elicited Rewards) for automated reward scoring, eliminating the need for manual reward function engineering and accelerating development by 2-3x.
*   **Easy Integration:**  Seamlessly integrate ART into your existing Python applications with an intuitive, ergonomic harness.
*   **Flexible Training:** Run ART on your local machine or utilize cloud-based GPU environments for optimized training efficiency.
*   **Observability & Debugging:** Integrate with popular platforms like W&B, Langfuse, and OpenPipe for comprehensive monitoring and streamlined debugging.
*   **Customizable:**  Tailor training parameters and inference engine configurations to meet specific project requirements, or leverage intelligent default settings optimized for performance and stability.

## RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by using an LLM as an automated judge to score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **RULER Benefits:**
*   **Faster Development:**  Reduce development time by skipping manual reward function engineering.
*   **Universal Applicability:** Works across diverse tasks without modification.
*   **High Performance:** Achieves performance parity or surpasses hand-crafted rewards in numerous benchmarks.
*   **Simple Implementation:**  Easy drop-in replacement for existing reward functions.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## Getting Started with ART

ART is an open-source RL framework, empowering LLMs to improve agent reliability through experience. ART provides an ergonomic harness for integrating GRPO into any python application.

### Quick Start: Train an Agent Now

Dive in with these interactive Colab notebooks. When you're ready to learn more, check out the [docs](https://art.openpipe.ai).

## 📒 Example Notebooks

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Choose ART?

*   **Simplified RL Integration:** Provides wrappers to easily integrate RL training into your existing applications.
*   **Flexible Deployment:** Run ART clients locally while leveraging cloud resources for training, or run everything locally.
*   **Enhanced Observability:** Integrate with popular platforms like W&B, Langfuse, and OpenPipe for comprehensive monitoring and streamlined debugging.
*   **Optimized Defaults:** Benefit from intelligent default configurations optimized for training efficiency and stability.

## Installation

Install the ART package within your project:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Check out the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post to learn how we trained Qwen 2.5 14B to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART operates with a **client-server** architecture, simplifying the training process:

1.  **Inference**
    *   Your code uses the ART client for agent workflows.
    *   Requests are routed to the ART server (running the model's latest LoRA in vLLM).
    *   Each `system`, `user`, and `assistant` message is stored in a Trajectory.
    *   Rollout completion triggers reward assignment.

2.  **Training**
    *   Trajectories are sent to the server for training. Inference is blocked.
    *   The server trains the model using GRPO, initializing from a checkpoint or an empty LoRA.
    *   The server saves the LoRA and loads it into vLLM.
    *   Inference resumes.

This loop continues until a defined number of iterations.

## 🧩 Supported Models

ART is designed to be compatible with a broad range of causal language models supported by vLLM/HuggingFace-transformers.  Check [Unsloth](https://docs.unsloth.ai/get-started/all-our-models) for supported models.  If you encounter issues with a specific model, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

ART thrives on community contributions!  Refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.

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

ART is built upon the contributions of many. We are especially grateful to the authors of the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners who've helped us test ART in the wild!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7

[**Back to the Top**](https://github.com/OpenPipe/ART)
```
Key improvements and SEO optimization:

*   **Clear, concise title:**  Includes the main keyword "Agent Reinforcement Trainer (ART)".
*   **One-sentence hook:** Immediately grabs the reader's attention and highlights the core value proposition.
*   **Bulleted key features:**  Makes the benefits easily scannable. Includes important SEO terms like "LLM agents," "Reinforcement Learning," "RULER," and "zero-shot".
*   **Keyword density:** The term "ART" and related terms are used naturally throughout the description.
*   **Clear section headers:** Uses `H2` and `H3` for better organization and readability.
*   **Calls to action:** Encourages users to try the notebooks, join the community, and contribute.
*   **Internal linking:** Provides links to documentation, and examples.
*   **External Links:** Keeps the external links like Discord & Github as original.
*   **Removed unnecessary repetition:** Removed duplicate phrases or overly verbose sentences.
*   **Concise and impactful language:**  Uses strong verbs and active voice to engage the reader.
*   **Back to top Link:** Easy navigation.