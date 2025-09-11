<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

</div>

**Unlock the power of AI agents with ART, an open-source framework that simplifies training LLMs to excel at real-world tasks using GRPO!**

[View the original repository on GitHub](https://github.com/OpenPipe/ART)

**Key Features:**

*   **Zero-Shot Reward Engineering with RULER:** Automate reward scoring using an LLM-as-judge.  No labeled data, expert feedback, or manual reward functions are needed!
*   **Accelerated Development:**  Reduce development time by eliminating the need for custom reward functions.
*   **General-Purpose Applicability:** Works across various tasks without modifications.
*   **Performance Driven:**  Achieves results that match or surpass hand-crafted reward systems in many benchmarks.
*   **Seamless Integration:**  Easy to integrate into existing applications, making it a drop-in replacement for traditional reward functions.
*   **Flexible Training:** Train agents from your laptop or leverage GPU-enabled environments.
*   **Enhanced Observability:** Integrates with W&B, Langfuse, and OpenPipe for improved debugging.
*   **Intelligent Defaults:** Utilize optimized defaults for training efficiency and stability.

[![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)](https://pypi.org/project/openpipe-art/)
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)
[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

## What is ART?

ART is an open-source reinforcement learning (RL) framework designed to enhance the reliability of AI agents by enabling LLMs to learn through experience. The framework provides a streamlined approach for integrating GRPO into Python applications.

### **RULER: Zero-Shot Agent Rewards**

**RULER** (**R**elative **U**niversal **L**LM-**E**licited **R**ewards) simplifies the agent training process by utilizing an LLM to automatically score agent trajectories. This approach eliminates the need for complex reward functions, offering a significant advantage in development speed and adaptability.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[Learn more about RULER](https://art.openpipe.ai/fundamentals/ruler)

## ART Notebooks: Get Started Quickly

Explore and train agents using provided notebooks:

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

## 📰 ART News & Updates

Stay informed on the latest advancements and research in the field:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[See all blog posts](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified RL Integration:** Easily incorporate RL training into existing applications.
*   **Flexible Deployment:** Train locally or leverage remote GPU environments.
*   **Enhanced Observability:** Benefit from integrations with platforms like W&B, Langfuse, and OpenPipe.
*   **Optimized Defaults:** Use intelligent defaults that ensure efficient and stable training.

## Installation

To add ART to your existing project:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent Example

Explore the practical application of ART in training an email agent that surpasses the capabilities of o3.

[Read the ART•E Agent blog post](https://openpipe.ai/blog/art-e-mail-agent)

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop

ART’s architecture is divided into a client and a server:

1.  **Inference:** The ART client drives the agent's actions, sending completion requests to the ART server which runs the latest LoRA in vLLM.
2.  **Reward and Training:** The client assigns a reward upon a rollout's completion. The server groups and trains on the trajectories, using GRPO to update the model, and then loads the updated LoRA into vLLM.

This process continues until the defined number of iterations is reached.

## 🧩 Supported Models

ART supports most vLLM/HuggingFace-transformers compatible causal language models (check [Unsloth](https://docs.unsloth.ai/get-started/all-our-models)). Report any compatibility issues on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues)!

## 🤝 Contributing

Contributions are welcome! Review the [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md) file for more details.

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

The source code is available under the [Apache-2.0 License](LICENSE).

## 🙏 Credits

ART is built with the help of these projects:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

Special thanks to our partners!