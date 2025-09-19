<div align="center">
  <a href="https://art.openpipe.ai">
    <picture>
      <img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
    </picture>
  </a>
  <h1>Agent Reinforcement Trainer (ART): Unleash the Power of LLMs with RL</h1>

  <p><b>Train advanced, multi-step agents for real-world tasks using Gradient-based Reinforcement Policy Optimization (GRPO) to boost reliability and performance.</b></p>

  [![PRs-Welcome](https://img.shields.io/badge/PRs-welcome-blue.svg)](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md)
  [![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)](https://pypi.org/project/openpipe-art/)
  [![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art-notebooks/blob/main/examples/2048/2048.ipynb)
  [![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
  [![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)
</div>

## Key Features of ART

*   **Zero-Shot Reward Engineering with RULER:** Leverage LLMs as judges to automatically score agent trajectories, eliminating the need for manual reward functions.
*   **Accelerated Development:** Reduce development time by up to 3x by skipping reward function engineering.
*   **General-Purpose Applicability:** Train agents across diverse tasks without task-specific modifications.
*   **Performance Enhancement:** Achieve or surpass the performance of agents trained with hand-crafted rewards in various benchmarks.
*   **Seamless Integration:** Easily incorporate ART into your existing Python applications.
*   **Modular Client-Server Architecture:** Train agents from any client machine and let the ART server handle the complexity of the training loop.
*   **Flexible Training Options:** Choose to train on a local GPU or an ephemeral GPU-enabled environment.
*   **Integration with Observability Platforms:** W&B, Langfuse, and OpenPipe integration simplifies debugging and enhances flexibility.

## RULER: Revolutionizing Reward Engineering

**RULER (Relative Universal LLM-Elicited Rewards)** simplifies reinforcement learning by utilizing an LLM to automatically evaluate agent behavior. This approach eliminates the need for manual reward function creation, labeled data, or expert feedback.

**Benefits of RULER:**

*   **Faster Development:** Significantly reduces the time spent on reward engineering.
*   **Versatile Application:** Works effectively across various tasks without modification.
*   **Competitive Performance:** Delivers performance that matches or exceeds hand-crafted rewards in a range of benchmarks.
*   **Simplified Integration:** Easily integrates into your existing projects.

```python
# Before: Complex Reward Function
def complex_reward_function(trajectory):
    # 50+ lines of scoring logic...
    pass

# After: RULER simplicity
judged_group = await ruler_score_group(group, "openai/o3")
```

[Learn more about RULER](https://art.openpipe.ai/fundamentals/ruler)

## ART Overview

ART is an open-source framework designed to improve agent reliability by enabling LLMs to learn from experience. The framework offers an efficient system for integrating Gradient-based Reinforcement Policy Optimization (GRPO) into Python applications.

## Examples & Notebooks

Get started quickly with these example notebooks demonstrating ART's capabilities:

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

## ART News

Stay informed with the latest advancements and updates in ART:

*   🗞️ **[ART now integrates seamlessly with LangGraph](https://art.openpipe.ai/integrations/langgraph-integration)** - Train your LangGraph agents with reinforcement learning for smarter multi-step reasoning and improved tool usage.
*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[See all blog posts](https://openpipe.ai/blog)

## Why Choose ART?

*   **Simplified Integration**: ART provides convenient wrappers for integrating RL training into existing applications, abstracting the training server for ease of use.
*   **Flexible Deployment**: Train agents locally or on a GPU-enabled cloud environment.
*   **Enhanced Observability**: Integrate with platforms like W&B, Langfuse, and OpenPipe for debugging and insights.
*   **Intelligent Defaults**: Benefit from optimized default training parameters and inference engine configurations.

## Installation

Easily install ART using pip:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent

Discover how ART can be applied in real-world scenarios by exploring the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent) blog post, showcasing the training of a Qwen 2.5 14B model that surpasses o3 in email retrieval.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 Training Loop Explained

ART operates through a client-server architecture:

1.  **Inference:**
    *   Your code utilizes the ART client to execute agentic workflows, often running rollouts in parallel.
    *   Requests are routed to the ART server, which executes the latest LoRA in vLLM.
    *   Messages are stored in a Trajectory.
    *   Rollouts are assigned a reward.
2.  **Training:**
    *   Trajectories are grouped and sent to the server for training.
    *   The server trains your model using GRPO, utilizing the latest checkpoint or an empty LoRA.
    *   The newly trained LoRA is saved and loaded into vLLM.
    *   Inference resumes.

This cycle repeats until the specified number of iterations is completed.

## 🧩 Supported Models

ART is designed to be compatible with most vLLM/HuggingFace-transformers compatible causal language models. If you encounter any compatibility issues, please reach out on [Discord](https://discord.gg/zbBHRUpwf4) or open an issue on [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contributing

We welcome contributions! For guidelines, see [CONTRIBUTING.md](https://github.com/openpipe/art/blob/main/CONTRIBUTING.md).

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

ART is built upon the work of many in the open-source RL community. Special thanks to the projects:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

Thank you to our partners for helping us test ART.

[Back to Top](#)
```

Key improvements:

*   **SEO Optimization:** Added clear headings, concise descriptions, and included relevant keywords (e.g., "Agent Reinforcement Trainer," "LLMs," "RL," "GRPO").
*   **Concise Summary:** The one-sentence hook at the beginning quickly grabs the reader's attention.
*   **Feature Highlighting:** Used bullet points to clearly list the key features and benefits.
*   **Improved Organization:** Structured the content logically with clear sections and subheadings.
*   **Actionable Information:** Kept the installation instructions prominent.
*   **Call to Action:** Encouraged users to explore the documentation and join the community.
*   **Internal Linking:** Added "[Back to Top](#)" at the bottom.
*   **Keyword Density:** Increased keyword usage in the headings and descriptions.
*   **Rephrased:** Better word choice to maximize clarity.