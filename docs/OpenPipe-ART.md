<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

**Train AI agents to master complex tasks with Agent Reinforcement Trainer (ART), an open-source framework that makes reinforcement learning for LLMs easier than ever.**  Find the original repository [here](https://github.com/OpenPipe/ART).

---

## Key Features of ART

*   **Effortless Reward Engineering with RULER:**  Eliminate the need for manual reward function creation. RULER (Relative Universal LLM-Elicited Rewards) automatically scores agent trajectories using an LLM-as-judge, saving you time and effort.
*   **Accelerated Development:**  Develop agents 2-3x faster by skipping the complex reward function engineering process.
*   **General-Purpose Applicability:**  ART is designed to work seamlessly across diverse tasks without modification.
*   **High Performance:**  Achieve or surpass the performance of hand-crafted reward functions in numerous benchmarks.
*   **Simple Integration:**  Easily integrate ART into your existing projects as a drop-in replacement for manual reward functions.
*   **Modular Architecture:**  ART is divided into a client and server, offering flexibility for your training setup. Train from anywhere!
*   **Comprehensive Support:**  Integrates with platforms such as W&B, Langfuse, and OpenPipe for detailed observability and efficient debugging.
*   **Customizable & Optimized:** Fine-tune your training parameters with intelligent defaults for efficient and stable training.

---

## Why Choose Agent Reinforcement Trainer (ART)?

ART offers a streamlined and efficient solution for training intelligent agents, enabling you to:

*   **Easily integrate RL into your applications:**  ART provides convenient wrappers to streamline the integration of RL training.
*   **Train from anywhere:** Run the ART client locally while leveraging a GPU-enabled environment for training, or use your own local GPU setup.
*   **Enhance Observability:**  Benefit from seamless integrations with platforms like W&B, Langfuse, and OpenPipe for detailed monitoring and efficient debugging.
*   **Leverage Optimized Defaults:**  Customize training parameters or utilize intelligent defaults optimized for training efficiency and stability.

---

## Getting Started

### Installation

Install ART with a single command:

```bash
pip install openpipe-art
```

### Notebook Examples

Explore these example notebooks to get started with ART:

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

---

## 🤖 ART•E Agent Example

See how ART can be applied in real-world tasks with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where we showcase how Qwen 2.5 14B was trained to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

---

## 🔁 Training Loop Explained

ART utilizes a client-server architecture for efficient training:

1.  **Inference:**
    *   Your code interacts with the ART client to perform agentic workflows.
    *   Requests are sent to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each message (`system`, `user`, `assistant`) is stored in a Trajectory.
    *   After a rollout, your code provides a `reward`.

2.  **Training:**
    *   Trajectories are grouped and sent to the server.
    *   The server trains the model using GRPO.
    *   The server saves the new LoRA and loads it into vLLM.
    *   Inference resumes.

This loop repeats until the desired training iterations are complete.

---

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models).  If you encounter issues with a specific model, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

---

## 🤝 Contributing

Contributions to ART are highly encouraged! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

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

## 🙏 Acknowledgements

ART is built upon the shoulders of giants.  We extend our gratitude to the open-source RL community, and specifically to the authors of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We're also thankful for the support from our partners!