<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  **Supercharge your LLM agents with ART, the open-source reinforcement learning framework for training multi-step agents.**
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Agent Reinforcement Trainer (ART): Train LLM Agents with Ease

ART is an open-source reinforcement learning (RL) framework designed to enhance the capabilities of LLM agents. ART simplifies the process of training multi-step agents for real-world tasks, allowing them to learn from experience and improve their performance.  [Learn more about ART](https://github.com/OpenPipe/ART).

### Key Features of ART:

*   **Zero-Shot Rewards with RULER:**  Leverage RULER (Relative Universal LLM-Elicited Rewards) to automatically score agent trajectories using an LLM-as-judge, eliminating the need for manual reward function engineering.
*   **Accelerated Development:** Significantly speed up development time by skipping reward function engineering.
*   **General-Purpose Applicability:** Train agents for any task without modification.
*   **Proven Performance:** Achieve performance that matches or exceeds hand-crafted rewards.
*   **Easy Integration:**  Drop-in replacement for manual reward functions.
*   **Modular Architecture:** Includes a client and server for flexible training from any machine.
*   **Customization:** Configure training parameters and inference engines.
*   **Comprehensive Integrations:** Seamlessly integrates with platforms such as W&B, Langfuse, and OpenPipe for observability and debugging.

###  RULER: Simplify Reward Engineering

**RULER** revolutionizes the reward function process by automatically scoring agent trajectories using an LLM-as-judge. Just define your task, and RULER takes care of the rest, eliminating the need for labeled data, expert feedback, or extensive reward engineering.

```python
# Before: Hours of reward engineering
def complex_reward_function(trajectory):
    # 50+ lines of careful scoring logic...
    pass

# After: One line with RULER
judged_group = await ruler_score_group(group, "openai/o3")
```

[📖 Learn more about RULER →](https://art.openpipe.ai/fundamentals/ruler)

###  Getting Started with ART

ART offers an ergonomic harness for integrating GRPO (Gradient Reinforcement Policy Optimization) into any Python application.

### 📒 Notebooks - Train Your Agent!

Explore the example notebooks below to get hands-on experience training agents for various tasks:

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

### 📰 Latest ART News and Updates

Stay up-to-date with the latest advancements and features in ART:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 See all blog posts →](https://openpipe.ai/blog)

### Why Choose ART?

ART offers key advantages for your agent training:

*   **Simplified Integration:** Easily introduce RL into existing applications.
*   **Flexible Training:** Run the ART client locally while utilizing an ephemeral GPU-enabled server environment.
*   **Enhanced Observability:** Integrations with platforms like W&B, Langfuse, and OpenPipe provide powerful debugging capabilities.
*   **Optimized Defaults:** Leverage intelligent defaults for efficient and stable training, or customize parameters to fit your specific requirements.

### Installation

Install ART agents from any client machine running Python:

```bash
pip install openpipe-art
```

### 🤖 ART•E Agent Example

Learn how to utilize ART for real-world tasks by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, where Qwen 2.5 14B was trained to outperform o3 in email retrieval!

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

### 🔁 Training Loop Explained

ART utilizes a client-server architecture.

**1. Inference:**

   1.  Your code uses the ART client to perform an agentic workflow.
   2.  Requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
   3.  Each `system`, `user`, and `assistant` message is stored in a Trajectory.
   4.  A `reward` is assigned to the Trajectory when a rollout is complete.

**2. Training:**
    1. Trajectories are grouped and sent to the server upon completion.
    2. The server trains your model using GRPO.
    3. The server saves the newly trained LoRA.
    4. Inference is unblocked and the loop resumes at step 1.

### 🧩 Supported Models

ART is designed to be compatible with most vLLM/HuggingFace-transformers-compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). Please report any issues on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

### 🤝 Contributing

Contributions to ART are warmly welcomed!  Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed information on how to contribute.

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

This repository's source code is available under the [Apache-2.0 License](LICENSE).

### 🙏 Acknowledgements

ART is built upon the work of numerous contributors within the open-source RL community. Special thanks to the developers of:

-   [Unsloth](https://github.com/unslothai/unsloth)
-   [vLLM](https://github.com/vllm-project/vllm)
-   [trl](https://github.com/huggingface/trl)
-   [torchtune](https://github.com/pytorch/torchtune)
-   [SkyPilot](https://github.com/skypilot-org/skypilot)

We're also grateful to our partners who have helped us test ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg