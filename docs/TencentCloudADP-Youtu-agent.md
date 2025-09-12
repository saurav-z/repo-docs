<!--
  This README.md is optimized for SEO, featuring clear headings, key features in bullet points,
  and a concise, compelling introduction. It's also designed for ease of navigation and provides
  specific, actionable information for users.

  Key improvements include:
  - Enhanced introduction with a strong hook and keyword integration.
  - Organized sections with clear headings for better readability and SEO.
  - Comprehensive feature listing, including benchmarks and automation.
  - Actionable "Getting Started" guide with Docker and Python setup.
  - Added links for deeper exploration and citation details.
  - Included links to core concepts, design philosophy, core features, and automation.
  - Added social proof (Star History chart).
-->

# 🤖 Youtu-Agent: Build Powerful Agents with Open-Source Models (**[GitHub Repo](https://github.com/TencentCloudADP/Youtu-agent)**)

Youtu-Agent is a cutting-edge agent framework designed to empower developers to create advanced, autonomous agents leveraging the power of open-source models for a variety of tasks.

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#-examples"><b>💡 Examples</b> </a>
| <a href="#-features"><b>✨ Features</b> </a>
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

## ✨ Key Features

*   **High-Performance Open-Source Agent Framework:** Achieve impressive results on benchmarks like WebWalkerQA and GAIA using open-source models.
*   **Optimized for Cost-Effective Deployment:** Designed to minimize costs, making agent development accessible without relying on closed models.
*   **Versatile Use Cases Out-of-the-Box:** Ready for tasks like data analysis, file processing, and research.
*   **Flexible and Extensible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting a variety of models, tools, and framework implementations.
*   **Automated Agent Creation:** Simplify setup with YAML configurations and automatic agent generation.

## 🚀 Getting Started

Quickly deploy and run your first agent using the steps below:

### 1. Setup

Ensure Python 3.12+ and [uv](https://github.com/astral-sh/uv) are installed.

Clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

Configure your `.env` file with the necessary API keys.

For DeepSeek API users, and to leverage free tokens:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

### 2. Quick Start

Run an interactive CLI chatbot with the default configuration:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

For Docker deployment, please refer to [`docker/README.md`](./docker/README.md).

### 3. Explore More Examples

Run the example to generate SVG images:

```bash
python examples/svg_generator/main.py
```

For the web UI (requires installation of the frontend package):

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

# Run the web version
python examples/svg_generator/main_web.py
```

## 🌟 Benchmark Performance

*   **WebWalkerQA:** Achieved **71.47%** accuracy using DeepSeek-V3.1, setting a new SOTA performance.
*   **GAIA:** Achieved **72.8%** pass@1 on the text-only validation subset using DeepSeek-V3-0324.

## 💡 Examples

Explore practical applications with these examples:

| Example                       | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| Data Analysis                 | Analyzes CSV files and generates HTML reports.    |
| File Management               | Renames and categorizes local files.              |
| Wide Research                 | Generates comprehensive reports.                 |
| Paper Analysis                | Parses and analyzes research papers.             |
| Automatic Agent Generation    | Simplifies agent configuration with YAML configs. |

## ✨ Features

### Core Features

*   Built on the [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   Fully asynchronous for efficient execution.
*   Tracing & analysis system (DBTracingProcessor).

### Automation

*   YAML-based configuration.
*   Automatic agent generation.
*   Tool generation & optimization (future support).

### Design Philosophy

*   Minimal design
*   Modular and Configurable
*   Open-source model support and low-cost deployment

### Use Cases

*   Deep / Wide research
*   Webpage generation
*   Trajectory collection

## 🤔 Why Choose Youtu-Agent?

*   **For Researchers:** A simple yet powerful baseline for LLM training.
*   **For Developers:** A proven framework for real-world agent applications.
*   **For Enthusiasts:** Practical examples and intuitive debugging tools.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and environment.
*   **Toolkit:** An encapsulated set of tools an agent can use.
*   **Environment:** The world the agent operates in (e.g., browser, shell).
*   **ContextManager:** A configurable module to manage the agent's context window.
*   **Benchmark:** An encapsulated workflow for datasets.

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

## 🙏 Acknowledgements

This project is built upon the foundations of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions. Review the [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

## 📚 Citation

If you use Youtu-Agent, please cite the project:

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

[![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)](https://star-history.com/#TencentCloudADP/youtu-agent)