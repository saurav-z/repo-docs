# Youtu-agent: Build Powerful Autonomous Agents with Open-Source Models

**Unlock the potential of open-source AI agents with Youtu-agent, a high-performance framework that simplifies building, running, and evaluating intelligent agents.** ([See the original repo](https://github.com/Tencent/Youtu-agent))

<div align="center">
<a href="https://tencent.github.io/Youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/Tencent/Youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#examples"><b>💡 Examples</b> </a>
| <a href="#features"><b>✨ Features</b> </a>
| <a href="#getting-started"><b>🚀 Getting Started</b> </a>
|
</p>

Youtu-agent empowers developers and researchers to create sophisticated AI agents for a variety of tasks, leveraging the power of open-source models and streamlining the development process.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Benefits:**

*   **High Performance:** Achieve state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1), utilizing open-source models.
*   **Cost-Effective & Open-Source Focused:** Built for accessibility and affordability, with a strong emphasis on open-source models and tools.
*   **Versatile Use Cases:**  Ready-to-use examples for data analysis, file management, research, and more.
*   **Flexible & Extensible:** Built on [openai-agents](https://github.com/openai/openai-agents-python), offering support for various model APIs, tool integrations, and framework implementations.
*   **Simplified Development:** YAML-based configuration, automated agent generation, and streamlined setup for faster development.

## 🌟 Benchmark Performance

Youtu-agent demonstrates impressive performance on challenging benchmarks, built using open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore practical applications of Youtu-agent with these example use cases:

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files for the user.
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775"
             poster="https://img.youtube.com/vi/SCR4Ru8_h5Q/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-427f-ba93-9ba21c5a579e"
             poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
  <tr >
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report, replicating the functionality of Manus.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a given paper, performs analysis, and compiles related literature to produce a final result.
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92"
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height=300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d"
             poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

### 🤖 Automatic Agent Generation

Youtu-agent simplifies agent creation with its automatic agent generation feature.  Define your agent's requirements through a simple conversation, and the framework will generate and save the configuration automatically using YAML.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define your agent's requirements and generate the configuration.
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding:10px; vertical-align:top; width: 400px;">
      <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef"
             poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg"
             controls muted preload="metadata"
             width="100%" height="auto"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

Explore more detailed examples and advanced use-cases in the [`examples`](./examples) directory and the [`docs/examples.md`](./docs/examples.md) documentation.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
*   **Minimal Design:** Keeps the framework simple and easy to use.
*   **Modular & Configurable:** Allows for flexible customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost:** Designed to be accessible and cost-effective.

### Core Features
*   **Built on openai-agents:** Leverages the openai-agents SDK for streaming, tracing, and agent loop capabilities.
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation
*   **YAML-Based Configuration:** Simplifies agent configuration management.
*   **Automatic Agent Generation:**  Generates agent configurations based on user input.
*   **Tool Generation & Optimization:** Automated tool evaluation, optimization, and customized tool generation will be supported in the future.

### Use Cases
*   **Deep / Wide Research:** Supports search-oriented tasks.
*   **Webpage Generation:**  Examples include generating web pages based on specific inputs.
*   **Trajectory Collection:**  Supports data collection for training and research.

## 🤔 Why Choose Youtu-agent?

Youtu-agent offers distinct advantages for various user groups:

### For Agents Researchers & LLM Trainers
*   A **strong baseline** for model training and ablation studies.
*   **One-click evaluation scripts** for streamlined benchmarking.

### For Agent Application Developers
*   A **reliable and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:**  Get started quickly with simple scripts and built-in toolkits.
*   **Modular Design:**  Customizable components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts
*   **Practical Use Cases:**  Examples in `/examples` for report generation, data analysis, and file organization.
*   **Simplicity & Debuggability:**  A rich toolset and visual tracing tools for intuitive development.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools for an agent.
*   **Environment:** The context in which the agent operates (e.g., browser, shell).
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** A workflow for a specific dataset, including preprocessing and judging.

For more details, refer to our [technical documentation](https://tencent.github.io/Youtu-agent/).

## 🚀 Getting Started

Follow these steps to quickly run your first agent:

### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Tencent/Youtu-agent.git
cd Youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
```

>   [!NOTE]
>   The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

### Quickstart

Use a built-in configuration, such as the default:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run the interactive CLI chatbot:

```bash
python scripts/cli_chat.py --stream --config default
```

📖 More details: [Quickstart Documentation](https://tencent.github.io/Youtu-agent/quickstart)

### Explore examples

Generate an SVG infographic based on a research topic:

```bash
python examples/svg_generator/main_web.py
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencent.github.io/Youtu-agent/examples)

### Run evaluations

Evaluate on `WebWalkerQA`:

```bash
# prepare dataset
python scripts/data/process_web_walker_qa.py
# run evaluation with config ww.yaml with your custom exp_id
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencent.github.io/Youtu-agent/eval)

## Acknowledgements

This project leverages the work of:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## Citation

```bibtex
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Tencent/Youtu-agent}},
}