# Youtu-Agent: Build Powerful Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework empowering you to build, run, and evaluate autonomous agents using open-source models, delivering exceptional results with a focus on accessibility and cost-effectiveness. Explore the [Youtu-Agent GitHub Repository](https://github.com/TencentCloudADP/youtu-agent) for more details.

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

## Key Features

*   **High Performance:** Achieves impressive benchmark scores on WebWalkerQA (71.47%) and GAIA (72.8% on text-only subset), using only DeepSeek-V3 series models.
*   **Open-Source & Cost-Effective:** Designed for accessible and affordable deployment, avoiding reliance on proprietary models.
*   **Versatile Use Cases:** Supports data analysis, file processing, literature review, and soon, podcast and video generation.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), with broad support for diverse model APIs (DeepSeek, GPT-OSS, etc.) and tool integrations.
*   **Automated Agent Creation:** YAML-based configurations, automated agent generation, and streamlined setup minimize manual effort.

## 🗞️ News

*   **Free DeepSeek Tokens:** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). Try it out and use DeepSeek models in `Youtu-Agent`!
*   **DeepSeek-V3.1 Update:** Check out the live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework.

## 🌟 Benchmark Performance

Youtu-Agent delivers strong performance on challenging benchmarks:

*   **WebWalkerQA:** Achieved 71.47% accuracy using DeepSeek-V3.1, setting a new state-of-the-art.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.  Multimodal tool evaluation is coming soon.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

See the power of Youtu-Agent in action:

| **Data Analysis**  | **File Management**     |
| :------------------ | :----------------------- |
| Analyzes a CSV and generates an HTML report. | Renames and categorizes local files.   |
| <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920"  poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| **Wide Research**  | **Paper Analysis**       |
| Gathers extensive information to generate a comprehensive report. | Parses and analyzes a paper. |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92"  poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation:

*   Interactively define requirements, and the "meta-agent" will generate and save the config.

```bash
# Interactively clarify requirements and auto-generate a config
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

Find more examples in the [`examples`](./examples) directory and detailed documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
*   **Minimal Design:** Keeps the framework simple and easy to use.
*   **Modular & Configurable:** Allows for flexible customization and integration of new components.
*   **Open-Source & Low-Cost:**  Focuses on accessibility and cost-effectiveness.

### Core Features
*   **Built on openai-agents:** Leverages the openai-agents SDK.
*   **Fully Asynchronous:** Provides high-performance execution.
*   **Tracing & Analysis System:**  The `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
*   **YAML-based Configuration:** Simplifies agent configuration management.
*   **Automatic Agent Generation:** Enables automatic configuration based on user requirements.
*   **Tool Generation & Optimization:** Automated tool generation and optimization will be supported in the future.

### Use Cases
*   **Deep / Wide Research:** Comprehensive search-oriented tasks.
*   **Webpage Generation:** Generate web pages.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers benefits for various user groups:

### For Agents Researchers & LLM Trainers
*   **Strong Baseline:** Serves as an excellent starting point for model training and research.
*   **One-Click Evaluation Scripts:** Streamlines the experimental process.

### For Agent Application Developers
*   **Proven & Portable:**  Scaffolding for building real-world agent applications.
*   **Ease of Use:** Get started with simple scripts and toolkits.
*   **Modular Design:**  Encapsulated and highly customizable components.

### For AI & Agent Enthusiasts
*   **Practical Use Cases:**  Examples for tasks like research, data analysis, and file organization.
*   **Simplicity & Debuggability:** Rich tools for development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** Encapsulated tools for an agent.
*   **Environment:** The environment where the agent operates.
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Encapsulated workflow for a specific dataset.

For more details, see our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

Clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

Populate `.env` with required API keys (LLM, etc.).

> **Important:**  Get **3 million free tokens** for the DeepSeek API from [Tencent Cloud International](https://www.tencentcloud.com/).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381) (**Sep 1 – Oct 31, 2025**)

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup.

### Quick Start

Run a default agent with a search tool using the CLI:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Example: Generate an SVG image based on a topic:

```bash
python examples/svg_generator/main.py
```

Or run the web UI version:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

# Run web version
python examples/svg_generator/main_web.py
```

Then access the local link in the terminal:

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on the WebWalkerQA dataset:

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

See [Evaluation Analysis](./frontend/exp_analysis/README.md) for results.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

## 🙏 Acknowledgements

This project uses the work of several open-source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Read our [**Contributing Guidelines**](./CONTRIBUTING.md) to help improve Youtu-Agent.

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