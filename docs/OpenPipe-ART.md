<div align="center">

<a href="https://art.openpipe.ai"><picture>
<img alt="ART logo" src="https://github.com/openpipe/art/raw/main/assets/ART_logo.png" width="160px">
</picture></a>

<p align="center">
  <h1>Agent Reinforcement Trainer (ART)</h1>
</p>

<p>
  **Supercharge your AI agents: ART empowers you to train advanced, multi-step agents for real-world tasks using the power of GRPO and reinforcement learning.**
</p>

[![PRs-Welcome][contribute-image]][contribute-url]
[![PyPI version](https://img.shields.io/pypi/v/openpipe-art?color=364fc7)][pypi-url]
[![Train Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)

[![Join Discord](https://img.shields.io/badge/Join%20Discord-5865F2?style=plastic&logo=discord&logoColor=white)](https://discord.gg/zbBHRUpwf4)
[![Documentation](https://img.shields.io/badge/Documentation-orange?style=plastic&logo=gitbook&logoColor=white)](https://art.openpipe.ai)

</div>

## Key Features of ART:

ART is a powerful open-source framework designed to enhance the reliability and performance of your AI agents through reinforcement learning.

*   **Effortless Integration**: Seamlessly integrate ART into your existing Python applications.
*   **No Labeled Data Required**: ART learns tasks by analyzing tools and their usage, eliminating the need for extensive labeled datasets.
*   **Versatile Application**: Optimize your agents for any Model Context Protocol (MCP) server or task.
*   **Strong Performance**: Achieve state-of-the-art results, matching or exceeding benchmarks in multiple scenarios.
*   **Modular Design**: Separate client and server components for flexible training and deployment.
*   **Customizable Training**: Fine-tune training parameters and inference engine configurations to meet specific needs.

## Train Your Agents with ART:

ART offers a streamlined approach to training AI agents for various complex tasks.  This section shows the power of the ART framework through a use case for training agents to master MCPs.

### MCP•RL: Teach Your Agents to Master MCP

ART shines with its **MCP•RL** capabilities, enabling you to train agents to effectively use any MCP (Model Context Protocol) server with minimal setup.  Simply provide a server URL, and ART will:

1.  **Automate Tool Discovery**: Automatically discover server tools.
2.  **Generate Input Tasks**: Design input tasks that effectively utilize those tools.
3.  **Optimize with RULER**: Train the model to improve performance on the MCP server using RULER (Reward function using Learned Evaluation and Ranking).
4.  **Validate Performance**: Test the trained model on new tasks to ensure optimal performance.

```python
from art.rewards import ruler_score_group

# Specialize a model for NWS MCP server
MCP_SERVER_URL = "https://server.smithery.ai/@smithery-ai/national-weather-service/mcp"

# Generate training scenarios based on MCP tools
scenarios = await generate_scenarios(
    num_scenarios=24,
    server_url=MCP_SERVER_URL,
)

# ...run the agent...

# Use RULER to assign relative scores to each trajectory
scored_groups = []
for group in groups:
    judged_group = await ruler_score_group(group)
    scored_groups.append(judged_group)

# Train the model to improve performance on the MCP server
await model.train(scored_groups)
```

## 💻 Getting Started: Notebook Examples

Dive into ART with our interactive notebooks. Each notebook provides a hands-on introduction to training agents for specific tasks. Explore and experiment!

| Agent Task         | Example Notebook                                                                                                             | Description                                     | Comparative Performance                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MCP•RL**         | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/mcp-rl/mcp-rl.ipynb)               | Qwen 2.5 3B masters the NWS MCP server          | [Link coming soon]                                                                                                                                                                                  |
| **ART•E [RULER]**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/art-e.ipynb)                       | Qwen 2.5 7B learns to search emails using RULER | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/email_agent/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/art-e/art_e/evaluate/display_benchmarks.ipynb) |
| **2048**           | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/2048/2048.ipynb)                   | Qwen 2.5 3B learns to play 2048                 | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/2048/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/2048/benchmark_2048.ipynb)                            |
| **Temporal Clue**  | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/temporal_clue/temporal-clue.ipynb) | Qwen 2.5 7B learns to solve Temporal Clue       | [Link coming soon]                                                                                                                                                                                  |
| **Tic Tac Toe**    | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/tic_tac_toe/tic-tac-toe.ipynb)     | Qwen 2.5 3B learns to play Tic Tac Toe          | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/tic-tac-toe-local/accuracy-training-progress.svg" height="72"> [benchmarks](/examples/tic_tac_toe/benchmark_tic_tac_toe.ipynb) |
| **Codenames**      | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/codenames/Codenames_RL.ipynb)      | Qwen 2.5 3B learns to play Codenames            | <img src="https://github.com/openpipe/art/raw/main/assets/benchmarks/codenames/win_rate_over_time.png" height="72"> [benchmarks](/examples/codenames/Codenames_RL.ipynb)                            |
| **AutoRL [RULER]** | [🏋️ Train agent](https://colab.research.google.com/github/openpipe/art/blob/main/examples/auto_rl.ipynb)                     | Train Qwen 2.5 7B to master any task            | [Link coming soon]                                                                                                                                                                                  |

## 📰 ART News & Updates

Stay informed about the latest advancements and applications of ART.

*   🗞️ **[MCP•RL: Teach Your Model to Master Any MCP Server](https://x.com/corbtt/status/1953171838382817625)** - Automatically train models to effectively use MCP server tools through reinforcement learning.
*   🗞️ **[AutoRL: Zero-Data Training for Any Task](https://x.com/mattshumer_/status/1950572449025650733)** - Train custom AI models without labeled data using automatic input generation and RULER evaluation.
*   🗞️ **[RULER: Easy Mode for RL Rewards](https://openpipe.ai/blog/ruler-easy-mode-for-rl-rewards)** is now available for automatic reward generation in reinforcement learning.
*   🗞️ **[ART·E: How We Built an Email Research Agent That Beats o3](https://openpipe.ai/blog/art-e-mail-agent)** demonstrates a Qwen 2.5 14B email agent outperforming OpenAI's o3.
*   🗞️ **[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer)** enables easy training of LLM-based agents using GRPO.

[📖 Explore all blog posts →](https://openpipe.ai/blog)

## Why Choose ART?

ART offers several advantages for building and training advanced AI agents:

*   **Seamless Integration**: ART provides easy-to-use wrappers to integrate RL training into existing applications.
*   **Flexible Training**: Train your agents from any location, leveraging local GPUs or cloud-based resources.
*   **Observability**: Benefit from integrations with platforms like W&B, Langfuse, and OpenPipe for streamlined debugging and monitoring.
*   **Intelligent Defaults**: Use optimized default settings or customize training parameters to align with your project requirements.

## 🚀 Installation

Easily integrate ART into your Python project with a simple pip command:

```bash
pip install openpipe-art
```

## 🤖 ART•E Agent: A Real-World Example

Discover how ART is used in practice with the [ART•E Agent](https://openpipe.ai/blog/art-e-mail-agent), where we trained Qwen 2.5 14B to excel at email retrieval, surpassing OpenAI's o3.

<img src="https://github.com/openpipe/art/raw/main/assets/ART_E_graphs.png" width="700">

## 🔁 The ART Training Loop: An Overview

ART employs a client-server architecture to streamline the RL training process:

1.  **Inference:**
    *   Your code utilizes the ART client to execute agentic workflows, often running several parallel rollouts for faster data collection.
    *   Completion requests are routed to the ART server, which runs the model's latest LoRA in vLLM.
    *   Each interaction (system, user, assistant) is stored in a Trajectory.
    *   At the rollout's end, your code assigns a `reward` to the Trajectory.

2.  **Training:**
    *   When all rollouts are done, Trajectories are grouped and sent to the server. Inference is blocked during training.
    *   The server trains your model using GRPO, starting from the latest checkpoint (or an empty LoRA initially).
    *   The server saves the updated LoRA and loads it into vLLM.
    *   Inference resumes.

This loop continues for a predetermined number of iterations.

## 🧩 Supported Models

ART is designed to work with most vLLM/HuggingFace-transformers compatible causal language models, including those supported by [Unsloth](https://docs.unsloth.ai/get-started/all-our-models). While Gemma 3 isn't currently supported, we encourage you to report any issues or questions on [Discord](https://discord.gg/zbBHRUpwf4) or [GitHub](https://github.com/openpipe/art/issues).

## 🤝 Contribute to ART

We welcome contributions to ART! For details, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

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

ART is licensed under the [Apache-2.0 License](LICENSE).

## 🙏 Acknowledgements

ART is built upon the work of numerous contributors in the open-source RL community. We are particularly grateful to the developers of:

*   [Unsloth](https://github.com/unslothai/unsloth)
*   [vLLM](https://github.com/vllm-project/vllm)
*   [trl](https://github.com/huggingface/trl)
*   [torchtune](https://github.com/pytorch/torchtune)
*   [SkyPilot](https://github.com/skypilot-org/skypilot)

We thank our partners for their support in testing and refining ART.

[pypi-url]: https://pypi.org/project/openpipe-art/
[contribute-url]: https://github.com/openpipe/art/blob/main/CONTRIBUTING.md
[contribute-image]: https://img.shields.io/badge/PRs-welcome-blue.svg
```
Key improvements and summaries:

*   **SEO Optimization:**  Keywords like "Agent Reinforcement Trainer," "AI agents," "reinforcement learning," and "GRPO" are strategically placed throughout the document.
*   **Compelling Hook:** The one-sentence hook at the beginning immediately grabs the reader's attention.
*   **Clear Headings and Structure:**  Uses clear headings and subheadings for readability and SEO benefits.
*   **Bulleted Key Features:**  Provides a concise summary of key features, improving scannability.
*   **Concise Descriptions:**  Replaces long paragraphs with shorter, more focused descriptions.
*   **Call to Action:** Includes a clear call to action (e.g., "Explore and experiment!") to encourage user engagement.
*   **Enhanced Formatting:** Uses bolding, code blocks, and images effectively to make the content visually appealing.
*   **Comprehensive Overview:** Provides a thorough explanation of the ART framework, including its benefits, installation, and usage examples.
*   **Internal Linking:** Includes links to the documentation, examples, and the original repository.
*   **Updated Content:** Kept the most important aspects and added some clarification and descriptions.
*   **Improved readability** The formatting makes it easy for the user to grasp the value of ART at a glance.