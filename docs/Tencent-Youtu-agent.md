# Youtu-agent: Build Powerful AI Agents with Open-Source Models

**Unleash the power of AI agents with Youtu-agent, a flexible and high-performance framework designed for building, running, and evaluating agents using open-source models. [(Original Repository)](https://github.com/Tencent/Youtu-agent)**

<div align="center">
<a href="https://tencentcloudadp.github.io/Youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/Youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/Youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#-examples"><b>💡 Examples</b> </a>
| <a href="#-features"><b>✨ Features</b> </a>
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a>
</p>

Youtu-agent empowers you to create intelligent agents capable of data analysis, file processing, and advanced research, all while leveraging the efficiency and cost-effectiveness of open-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   **High Performance:** Achieved impressive results on WebWalkerQA (71.47% pass@1) and GAIA (72.8% on the text-only subset, pass@1) using DeepSeek-V3 models.
*   **Cost-Effective & Open-Source Focused:** Designed to run efficiently on open-source models, reducing reliance on expensive closed-source alternatives.
*   **Practical Use Cases:** Built-in support for tasks like CSV analysis, literature review, file organization, and more.
*   **Flexible & Extensible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python) with support for diverse model APIs, tool integrations, and custom frameworks.
*   **Simplified Development:** Utilize YAML-based configurations, auto agent generation, and streamlined setup for rapid prototyping and deployment.

## ✨ Key Highlights

*   **Performance Leaderboard**: Showcases impressive results on benchmark datasets.
*   **Automatic Agent Generation:** Significantly simplifies agent configuration with user-friendly YAML-based configs.
*   **Rich Toolset**: Includes CSV analysis, file organization, and deep research capabilities.

## 🌟 Benchmark Performance

Youtu-agent demonstrates strong results on deep search and tool use benchmarks:

*   **WebWalkerQA:** Achieved a new SOTA performance of 71.47% using DeepSeek-V3.1.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.

## 💡 Examples

Explore the capabilities of Youtu-agent through these interactive examples:

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
      <video src="https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775"
             poster="https://img.youtube.com/vi/SCR4Ru8_h5Q/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files for the user.
      <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e"
             poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
  <tr >
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report, replicating the functionality of Manus.
      <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92"
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height=300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a given paper, performs analysis, and compiles related literature to produce a final result.
      <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d"
             poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

### 🤖 Automatic Agent Generation

Simplify your workflow with Youtu-agent's automatic agent generation feature:

*   **Interactive Configuration:**  Specify your requirements, and the system will automatically create and save the agent configuration based on a simple YAML file.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively clarify your requirements, automatically generate the agent configuration, and run it right away.
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

Explore the full range of examples in the [`examples`](./examples) directory and dive into our detailed documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeps the framework simple and efficient.
*   **Modular & Configurable:** Enables flexible customization and integration of new components.
*   **Open-Source & Cost-Effective:** Promotes accessibility and lowers costs for various applications.

### Core Features

*   **Based on openai-agents:** Leveraging openai-agents SDK.
*   **Fully Asynchronous:** Ensures high-performance and efficient execution.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories (coming soon).

### Automation

*   **YAML-Based Configuration:** Simplifies agent management with structured configurations.
*   **Automatic Agent Generation:** Allows agent configurations to be generated based on user requirements.
*   **Tool Generation & Optimization:** Supports tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep/Wide Research:** Supports common search-oriented tasks.
*   **Webpage Generation:** Generates web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-agent?

Youtu-agent is designed for:

### Agent Researchers & LLM Trainers

*   **Baseline for LLM Training:** Strong baseline for model training and ablation studies.
*   **Streamlined Benchmarking:** One-click evaluation scripts for consistent and efficient experimentation.

### Agent Application Developers

*   **Proven & Portable Framework:** A scaffolding for building real-world agent applications.
*   **Ease of Use:** Quick start with simple scripts and built-in toolkits.
*   **Modular Design:** Encapsulated and highly customizable components.

### AI & Agent Enthusiasts

*   **Practical Examples:**  Tasks such as deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A rich toolset and visual tracing for straightforward development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more details, please refer to our [technical documentation](https://tencentcloudadp.github.io/Youtu-agent/).

## 🚀 Getting Started

Follow these steps to run your first agent:

### Setup

```bash
git clone https://github.com/TencentCloudADP/Youtu-agent.git
cd Youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # config necessary keys...
```

### Quickstart

Run an interactive CLI chatbot with the default search tool:

```bash
python scripts/cli_chat.py --stream --config default
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/Youtu-agent/quickstart)

### Explore Examples

Generate an SVG infographic based on a research topic:

```bash
python examples/svg_generator/main_web.py
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/Youtu-agent/examples)

### Run Evaluations

Evaluate on `WebWalkerQA`:

```bash
# prepare dataset
python scripts/data/process_web_walker_qa.py
# run evaluation with config ww.yaml with your custom exp_id
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/Youtu-agent/eval)

## 🙏 Acknowledgements

This project is built upon the contributions of these open-source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

```bibtex
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/Youtu-agent}},
}
```

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/Youtu-agent&type=Date)