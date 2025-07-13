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

## Agent Reinforcement Trainer (ART): Effortlessly Train LLM Agents

**ART (Agent Reinforcement Trainer) empowers you to train powerful LLM agents for complex tasks using Reinforcement Learning from human preferences (GRPO), enabling you to bypass tedious reward function engineering and accelerate agent development.**  Learn more about ART on the [original GitHub repository](https://github.com/OpenPipe/ART).

### Key Features:

*   **Zero-Shot Reward Functions with RULER:**  Leverage the power of LLMs to automatically score agent trajectories with **RULER** (Relative Universal LLM-Elicited Rewards) eliminating the need for manual reward function creation and speeding up development.
*   **Accelerated Development:**  Develop agents 2-3x faster by skipping reward function engineering.
*   **General Applicability:** Train agents across any task without modification.
*   **High Performance:** Achieve results that match or surpass hand-crafted reward functions in many benchmarks.
*   **Simplified Integration:** Seamlessly integrate ART into your existing applications.
*   **Flexible Training Environment:** Run ART client on your laptop or leverage ephemeral GPU-enabled environments for training, supporting both local and cloud-based GPU usage.
*   **Enhanced Observability:** Integrate with platforms like Weights & Biases (W&B), Langfuse, and OpenPipe for simplified debugging and monitoring.
*   **Customizable & Optimized:**  ART provides intelligent defaults while allowing you to customize training parameters and inference engine configurations.

### RULER: Zero-Shot Agent Rewards

**RULER** (Relative Universal LLM-Elicited Rewards) eliminates the need for hand-crafted reward functions by using an LLM-as-judge to automatically score agent trajectories. Simply define your task in the system prompt, and RULER handles the rest—**no labeled data, expert feedback, or reward engineering required**.

✨ **Key Benefits:**
- **2-3x faster development** - Skip reward function engineering entirely
- **General-purpose** - Works across any task without modification
- **Strong performance** - Matches or exceeds hand-crafted rewards in 3/4 benchmarks
- **Easy integration** - Drop-in replacement for manual reward functions

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

### Example Notebooks

Get started quickly with these interactive examples:

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

### Why Choose ART?

*   **Seamless Integration:** Easily integrate RL into existing applications.
*   **Flexible Training:** Train agents locally or in the cloud.
*   **Comprehensive Observability:** Monitor and debug training with platform integrations.
*   **Optimized Defaults:** Leverage intelligent defaults or customize as needed.

### Installation

Install ART with a single command:

```bash
pip install openpipe-art
```

### 🤖 ART•E Agent

Explore a real-world application of ART with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where Qwen 2.5 14B was trained to excel at email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

### Training Loop Overview

ART's architecture divides functionality into a **client** and a **server**:

1.  **Inference:**
    *   Your code uses the ART client to perform agentic workflows.
    *   Completion requests are routed to the ART server, running the model's latest LoRA.
    *   Each message (`system`, `user`, `assistant`) is stored in a Trajectory.
    *   When a rollout finishes, your code assigns a `reward` to its Trajectory.
2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains your model using GRPO.
    *   The server saves the new LoRA and loads it into vLLM.
    *   Inference is unblocked, and the loop restarts.

### 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models. Contact us on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues) if you have trouble with specific models.

### 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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

Available under the [Apache-2.0 License](LICENSE).

### 🙏 Credits

Special thanks to the authors of [Unsloth](https://github.com/unslothai/unsloth), [vLLM](https://github.com/vllm-project/vllm), [trl](https://github.com/huggingface/trl), [torchtune](https://github.com/pytorch/torchtune), and [SkyPilot](https://github.com/skypilot-org/skypilot) for their contributions. Thank you to our partners for testing ART!

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
[downloads-image]: https://img.shields.io/pypi/dm/openpipe-art?color=364fc7&logoColor=364fc7