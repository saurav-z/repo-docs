# 🤖 Youtu-Agent: Build and Deploy Powerful AI Agents with Open-Source Models

**Youtu-Agent empowers you to effortlessly create and deploy AI agents with state-of-the-art performance using open-source models.** ([Original Repository](https://github.com/Tencent/Youtu-agent))

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

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

## Key Features

*   **High Performance & Open Source Focus**: Achieve state-of-the-art results using open-source models like DeepSeek-V3.
*   **Cost-Effective**: Designed for accessible, low-cost deployment, avoiding reliance on expensive, closed models.
*   **Versatile Use Cases**: Supports diverse tasks including data analysis, file processing, and content generation.
*   **Flexible Architecture**: Built on [openai-agents](https://github.com/openai/openai-agents-python), with extensibility for various models, tools, and frameworks.
*   **Simplified Development**: YAML-based configurations and automated agent generation streamline setup and reduce manual effort.

## 🚀 Getting Started

Follow these steps to get started with Youtu-Agent:

### 1. Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

First, make sure Python and uv are installed.

Then clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

After copying the `.env.example` file, you need to fill in the necessary keys in the `.env` file, e.g. LLM API keys. For example:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Once you’ve applied, replace the API key in the .env file below:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Please refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with interactive frontend.

### 2. Quick Start

Youtu-agent ships with built-in configurations. For example, the config `configs/agents/simple/base_search.yaml` defines a simple agent equipped with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

You can launch an interactive CLI chatbot with this agent by running:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### 3. Explore More Examples

The repository provides multiple ready-to-use examples. Some examples require the agent to have internet search capabilities, so you’ll need to configure the tool APIs in the `.env` file under the tools module:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

For example, to enable the agent to automatically search online for information and generate an SVG image on the topic of “DeepSeek V3.1 New Features,” run the following command:

```bash
python examples/svg_generator/main.py
```

If you want to visualize the agent’s runtime status using the web UI, download the frontend package from the Youtu-Agent releases and install it locally:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Next, run the web version of the SVG image generation command:

```bash
python examples/svg_generator/main_web.py
```

Once the terminal shows the following message, the deployment is successful. You can access the project by clicking the local link:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

Given a research topic, the agent will automatically search the web, collect relevant information, and output an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### 4. Run Evaluations

Youtu-Agent also supports benchmarking on standard datasets. For example, to evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and can be further analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## ✨ Features

### Design Philosophy

*   **Minimal Design**: Simple and easy-to-use framework.
*   **Modular and Configurable**: Flexibility for customization and integration.
*   **Open-Source Model Support & Low Cost**: Accessible and cost-effective for various applications.

### Core Features

*   **Built on openai-agents**: Leveraging the foundations of the [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   **Fully Asynchronous**: Enables high-performance and efficient execution.
*   **Tracing & Analysis System**: In-depth analysis of tool calls and agent trajectories.

### Automation

*   **YAML-Based Configuration**: Structured and manageable agent configurations.
*   **Automatic Agent Generation**: Generate configurations based on user requirements.
*   **Tool Generation & Optimization**: Tool evaluation and automated optimization.

### Use Cases

*   **Deep/Wide Research**: Supports common search-oriented tasks.
*   **Webpage Generation**: Generate web pages based on specific inputs.
*   **Trajectory Collection**: Supports data collection for training and research.

## 🌟 Benchmark Performance

`Youtu-Agent` excels in challenging deep search and tool use benchmarks:

*   **WebWalkerQA**: Achieved 71.47% using DeepSeek-V3.1.
*   **GAIA**: Achieved 72.8% pass@1 on the text-only subset using DeepSeek-V3-0324.

## 🤔 Why Choose Youtu-Agent?

*   **For Researchers**:  Simple, powerful baseline for model training and benchmarking.
*   **For Developers**: Proven scaffolding for building real-world agent applications.
*   **For Enthusiasts**: Practical examples, simplicity, and intuitive debugging tools.

## 🧩 Core Concepts

*   **Agent**: LLM with prompts, tools, and an environment.
*   **Toolkit**: Encapsulated set of tools for an agent.
*   **Environment**: The world the agent operates in.
*   **ContextManager**: Manages the agent's context window.
*   **Benchmark**: Encapsulated workflow for a specific dataset.

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts and features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get up and running quickly.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the excellent work of several open-source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions! See our [**Contributing Guidelines**](./CONTRIBUTING.md).

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