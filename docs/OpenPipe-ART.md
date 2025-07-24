<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

ART empowers you to train sophisticated multi-step agents with LLMs, quickly and easily using GRPO, an open-source RL framework.  [**Check out the original repo!**](https://github.com/OpenPipe/ART)

## Key Features

*   **Zero-Shot Rewards with RULER:** Automate agent evaluation with Relative Universal LLM-Elicited Rewards (RULER), eliminating the need for manual reward function engineering.
*   **Simplified Development:** Significantly accelerate development by skipping the complexities of reward function engineering.
*   **General-Purpose Applicability:** ART is designed to work across diverse tasks without modification.
*   **High Performance:** Achieve performance that matches or exceeds hand-crafted reward systems in several benchmarks.
*   **Easy Integration:** Seamlessly integrate ART into your projects as a drop-in replacement for manual reward functions.
*   **Modular Architecture:** ART's architecture separates the client and server, allowing you to run the client on your local machine and take advantage of a GPU enabled server.
*   **Flexible Training:** Run the ART client on your laptop and let the ART server kick off an ephemeral GPU-enabled environment, or run on a local GPU.
*   **Enhanced Observability:** Integrate with hosted platforms like W&B, Langfuse, and OpenPipe for enhanced debugging and monitoring.
*   **Customizable & Efficient:** Leverage intelligent defaults, optimize training parameters, and customize inference engine configurations to align with your specific needs.

## RULER: Automate Agent Scoring

**RULER** (Relative Universal LLM-Elicited Rewards) revolutionizes agent training by employing an LLM-as-judge to automatically score agent trajectories.  Just define your task in the system prompt, and RULER handles the rest.  This approach bypasses the need for labeled data, expert feedback, or complex reward engineering.

**Key Benefits of RULER:**

*   **Faster Development:** Reduce development time by a factor of 2-3x by eliminating reward function engineering.
*   **Universal Applicability:** Works across any task without modification.
*   **Competitive Performance:** Matches or surpasses hand-crafted reward systems in 3 out of 4 benchmarks.
*   **Simple Integration:** A straightforward drop-in replacement for manual reward functions.

```python
# Before: Extensive reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART is an open-source RL framework that trains agents by enabling LLMs to **learn from experience**.  It provides an ergonomic harness for integrating GRPO into any Python application. Explore the notebooks below for a hands-on introduction or dive into the [docs](https://art.openpipe.ai) for more in-depth information.

## Example Notebooks

| Agent Task        | Example Notebook                                                                                                             | Description                               | Comparative Performance                                                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ART•E [RULER]**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e/art-e.ipynb)               | Qwen 2.5 7B learns to search emails using RULER     | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb)                                                                                                                                          |
| **2048**          | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048           | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue | [Link coming soon]                                                                                                                                          |
| **Tic Tac Toe**   | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe    | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**     | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames      | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |

## Why Use ART?

*   **Seamless Integration:** Quickly integrate RL training into your existing applications with ART's convenient wrappers.
*   **Flexible Training:** Train from virtually anywhere, leveraging local GPUs or the ART server's GPU-enabled environments.
*   **Simplified Debugging:** Benefit from integrations with W&B, Langfuse, and OpenPipe for enhanced observability.
*   **Optimized Defaults:** Get started quickly with ART's intelligent defaults, which are fine-tuned for training efficiency and stability.

## Installation

To install ART, run the following command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Learn about the ART•E Agent and see how we trained Qwen 2.5 14B to excel at email retrieval in this detailed blog post: [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent).

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Overview

ART's architecture divides functionality between a **client** and a **server**.

1.  **Inference (Client)**
    1.  Your code uses the ART client to perform an agentic workflow, often executing several rollouts in parallel.
    2.  Completion requests are routed to the ART server, which runs the latest LoRA in vLLM.
    3.  Each message (`system`, `user`, `assistant`) is stored in a Trajectory.
    4.  When a rollout finishes, your code assigns a `reward` to the Trajectory.
2.  **Training (Server)**
    1.  Once each rollout is finished, Trajectories are grouped and sent to the server. Inference is blocked during training.
    2.  The server trains your model using GRPO, starting from the latest checkpoint or an empty LoRA.
    3.  The server saves the trained LoRA and loads it into vLLM.
    4.  Inference is unblocked, and the loop resumes at step 1.

This training loop continues until the specified number of inference and training iterations are complete.

## 🧩 Supported Models

ART is compatible with most vLLM/HuggingFace-transformers causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Please report any model compatibility issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

Contributions to ART are greatly appreciated. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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

ART is inspired by the open-source RL community. Special thanks to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We are grateful to our partners for their support in testing ART!