<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>

  <h1>Agent Reinforcement Trainer (ART)</h1>

  <p>Train intelligent agents for real-world tasks with ease using ART and its revolutionary <a href="#ruler-zero-shot-agent-rewards">RULER</a> technology.</p>

  [![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
  [![Downloads](https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7)](https://pypi.org/project/openpipe-art/)
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features of ART:

*   🚀 **Effortless Agent Training:** Streamline the process of training multi-step agents for diverse real-world tasks.
*   🧠 **GRPO-Powered:** Leverage the power of Gradient-Based Policy Optimization for robust and efficient training.
*   💻 **Open-Source & Flexible:** An open-source RL framework that easily integrates with existing Python applications.
*   ☁️ **Flexible Deployment:** Run training locally or in an ephemeral GPU-enabled environment, ideal for various needs.
*   🛠️ **Customizable:** Configure training parameters and leverage intelligent defaults, optimized for efficiency and stability.
*   📊 **Integrations:** Integrates with platforms like W&B, Langfuse, and OpenPipe for enhanced observability and debugging.
*   📦 **Easy Installation:** Simple pip installation to quickly add ART to your project.
*   📚 **Comprehensive Examples:** Get started with notebooks showcasing agent training for various tasks.

---

## 📏 RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by eliminating the need for hand-crafted reward functions.  RULER uses an LLM to automatically score agent trajectories, requiring **no labeled data, expert feedback, or manual reward engineering.**

**Key Benefits:**

*   **2-3x Faster Development:** Skip reward function engineering entirely.
*   **General-Purpose:** Adaptable across any task without modification.
*   **Strong Performance:** Matches or surpasses hand-crafted rewards in most benchmarks.
*   **Easy Integration:** Drop-in replacement for your manual reward functions.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

---

## ART Overview

ART is an open-source RL framework that enhances agent reliability by enabling LLMs to learn from experience. It provides an ergonomic harness for integrating GRPO into any python application. Dive into the example notebooks below for a hands-on introduction, and explore the [docs](https://art.openpipe.ai) for in-depth learning.

---

## 📒 Example Notebooks

Quickly get started with ART by training agents on the following tasks.

| Agent Task        | Example Notebook                                                                                                             | Description                                         | Comparative Performance                                                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue          | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe             | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames               | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

---

## Why Choose ART?

*   **Easy Integration:** Seamlessly incorporate RL training into existing applications with convenient wrappers.
*   **Flexible Training:** Train agents locally or in the cloud using ephemeral GPU environments.
*   **Enhanced Observability:** Integrations with popular platforms like W&B, Langfuse, and OpenPipe for flexible observability.
*   **Intelligent Defaults:** Benefit from pre-optimized configurations for efficient and stable training.

---

## Installation

Install the ART package directly into your Python project:

```bash
pip install openpipe-art
```

---

## 🤖 ART•E Agent

Explore a real-world application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where a Qwen 2.5 14B agent was trained to beat o3 at email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

---

## 🔁 Training Loop Overview

ART's training loop is divided into a **client** and a **server**, allowing for easy integration with your existing code.

1.  **Inference:**

    *   Your code uses the ART client to perform agentic workflows, gathering data through parallel rollouts.
    *   Completion requests are routed to the ART server.
    *   Each message (`system`, `user`, `assistant`) is stored in a Trajectory.
    *   Upon rollout completion, your code assigns a reward to the Trajectory.

2.  **Training:**

    *   Finished Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO, starting from the latest checkpoint (or a new LoRA).
    *   The newly trained LoRA is saved and loaded into vLLM.
    *   Inference resumes, and the loop repeats.

This loop continues until a set number of iterations are completed.

---

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models. For compatibility questions, see [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  Reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues) if you experience issues with your model.

---

## 🤝 Contributing

We welcome contributions! Learn how to contribute by reviewing the [CONTRIBUTING.md](CONTRIBUTING.md) file.

---

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

---

## ⚖️ License

This project is licensed under the [Apache-2.0 License](LICENSE).

---

## 🙏 Credits

ART is built upon the hard work of the open-source RL community. We extend our gratitude to the following projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for testing ART in the wild! We're excited to see what you all build with it.

---

<a href="https://github.com/OpenPipe/ART">Back to Top</a>
```
Key improvements and SEO considerations:

*   **Strong Headline and Hook:** Starts with a clear statement of what the project does and highlights the key benefit (RULER).
*   **Keyword Optimization:** Uses relevant keywords like "Agent Reinforcement Trainer," "RL," "LLM," and "GRPO."
*   **Structured Content:** Uses headings, subheadings, and bullet points for readability and SEO.
*   **Focus on Benefits:** Emphasizes the advantages of using ART and RULER (speed, performance, ease of use).
*   **Internal Linking:** Includes links to the project documentation and example notebooks.
*   **External Linking:**  Links to the original repo, Discord, and other relevant resources.
*   **Clear Call to Action:**  Encourages users to try the example notebooks and contribute.
*   **Concise Summarization:** Provides a brief but comprehensive overview of the project's features and functionality.
*   **Mobile-Friendly:** Uses HTML to make it more readable across different devices.
*   **Back to Top:** Includes a "Back to Top" link to enable quick navigation.