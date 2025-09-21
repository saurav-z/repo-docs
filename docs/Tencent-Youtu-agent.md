# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance agent framework, enabling developers to create and deploy sophisticated AI agents using open-source models. [Explore the Youtu-Agent Repository](https://github.com/Tencent/Youtu-agent)

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a> 
| <a href="#-examples"><b>💡 Examples</b> </a> 
| <a href="#-features"><b>✨ Features</b> </a> 
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a> 
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a> 
</p>

[![Youtu-Agent Logo](docs/assets/mascot.png)](https://github.com/Tencent/Youtu-agent)

## ✨ Key Features

*   ✅ **Open-Source Focused:** Designed for cost-effective deployment with open-source models.
*   🚀 **High Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%).
*   🛠️ **Flexible & Extensible:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting diverse model APIs, tool integrations, and framework implementations.
*   🤖 **Automated Agent Generation:** Streamlined YAML-based configuration and automatic agent generation simplify setup.
*   💡 **Practical Use Cases:** Supports CSV analysis, file processing, and deep research tasks out-of-the-box.

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and tools to deliver impressive results on challenging benchmarks.

*   **WebWalkerQA:** Achieved 71.47% accuracy using DeepSeek-V3.1, setting a new state-of-the-art.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only subset using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore various agent applications with interactive demos.

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
      <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920" 
             poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
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
             width="100%" height="300"
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

> [!NOTE]
> See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation.

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

> [!NOTE]
> See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
-   **Minimal design**: Keeps the framework simple and easy to use.
-   **Modular & configurable**: Enables flexible customization and easy integration.
-   **Open-source model support & low-cost**: Promotes accessibility and cost-effectiveness.

### Core Features
-   **Built on openai-agents**: Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python).
-   **Fully asynchronous**: Enables high-performance and efficient execution.
-   **Tracing & analysis system**: Provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
-   **YAML based configuration**: Structured and easily manageable agent configurations.
-   **Automatic agent generation**: Based on user requirements, agent configurations can be automatically generated.
-   **Tool generation & optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases
-   **Deep / Wide research**: Covers common search-oriented tasks.
-   **Webpage generation**: Examples include generating web pages based on specific inputs.
-   **Trajectory collection**: Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

*   **For Researchers:** A strong baseline for model training and ablation studies with one-click evaluation.
*   **For Developers:** A proven scaffolding for building real-world agent applications with ease of use and modular design.
*   **For Enthusiasts:** Practical use cases and a rich toolset for intuitive development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more design and implementation details, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to get started with Youtu-Agent.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Install Python and uv:** Ensure Python and uv are installed.
2.  **Clone the Repository:** `git clone https://github.com/TencentCloudADP/youtu-agent.git`
3.  **Navigate to the Directory:** `cd youtu-agent`
4.  **Install Dependencies:** `uv sync` (or `make sync`)
5.  **Activate Virtual Environment:** `source ./.venv/bin/activate`
6.  **Configure .env:** `cp .env.example .env` and fill in your API keys.  See the original for the sample configuration information.

#### Docker Deployment

Please refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with interactive frontend.

### Quick Start

Run an interactive CLI chatbot with a search tool:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Example: Generate an SVG image based on a search:

```bash
python examples/svg_generator/main.py
```

To visualize with the web UI:

1.  Download the frontend package.
2.  Install: `uv pip install utu_agent_ui-0.2.0-py3-none-any.whl`
3.  Run the web version: `python examples/svg_generator/main_web.py`

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Example: Evaluate on WebWalkerQA.

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Read our [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute.

## 📚 Citation

```bibtex
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)