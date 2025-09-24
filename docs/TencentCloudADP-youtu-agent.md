# Youtu-Agent: Build Powerful Agents with Open-Source Models

> **Youtu-Agent is a flexible and efficient framework for building and deploying autonomous agents, achieving impressive results with open-source models like DeepSeek-V3.**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

[中文](README_ZH.md) | [日本語](README_JA.md) | [Performance](#-benchmark-performance) | [Examples](#-examples) | [Features](#-features) | [Getting Started](#-getting-started) | [Join Community](https://discord.gg/svwuqgUx)

Youtu-Agent is a high-performance, open-source framework designed to simplify the creation, execution, and evaluation of autonomous agents. It empowers developers to build agents for a wide range of tasks, from data analysis to complex research, all while leveraging the power of open-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   **High Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1 text-only subset) using only DeepSeek-V3 models, showcasing the power of open-source solutions.
*   **Cost-Effective & Open-Source Focused:** Designed for accessible and affordable deployment, avoiding reliance on expensive closed-source models.
*   **Practical Use Cases:** Includes out-of-the-box support for data analysis, file processing, literature reviews, and more, with podcast/video generation coming soon.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), ensuring broad compatibility with various model APIs (DeepSeek, gpt-oss, etc.) and tool integrations.
*   **Automation & Simplicity:** YAML-based configurations, automated agent generation, and streamlined setup to reduce development overhead.

## News

*   📺 **September 9, 2025:** Live sharing of design philosophy and basic usage of Youtu-Agent. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)]
*   🎁 **September 2, 2025:** Tencent Cloud International offers new DeepSeek API users **3 million free tokens** (September 1 – October 31, 2025). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. For enterprise agent solutions, check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **August 28, 2025:** Live sharing of updates about DeepSeek-V3.1 and how to use it in the Youtu-Agent framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)]

## 🌟 Benchmark Performance

Youtu-Agent excels in benchmarks, demonstrating strong results with open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA):** Achieved 71.47% accuracy with DeepSeek-V3.1, establishing a new state-of-the-art performance.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/):** Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using DeepSeek-V3-0324. Evaluation is ongoing on the full GAIA benchmark with multimodal tools.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the capabilities of Youtu-Agent with these interactive examples. Click the images to view detailed videos.

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyze CSV files and generate HTML reports.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Rename and categorize local files.
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
      <strong>Wide Research</strong><br>Generate comprehensive reports by gathering extensive information.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyze papers, perform literature reviews, and compile results.
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
>  See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more detailed examples.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic agent generation.  Create agents without writing code using simple YAML-based configurations.  A built-in "meta-agent" guides you to define requirements and then auto-generates and saves the config.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Specify your needs interactively, then automatically generate and run the agent configuration.
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
>  See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** The framework prioritizes simplicity and ease of use, minimizing unnecessary complexity.
*   **Modular & Configurable:** Provides flexibility for customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost:** Enables accessible and cost-effective solutions for various applications.

### Core Features

*   **Built on openai-agents:**  Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python), ensuring compatibility with `responses` and `chat.completions` APIs for adaptation to models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:**  Enables high-performance and efficient execution, especially beneficial for benchmark evaluations.
*   **Tracing & Analysis System:** `DBTracingProcessor` provides in-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation

*   **YAML-based Configuration:**  Manage agent configurations with structured and easily modifiable files.
*   **Automatic Agent Generation:**  Generate agent configurations automatically based on user requirements.
*   **Tool Generation & Optimization:**  Future support for tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep/Wide Research:**  Supports common search-oriented tasks.
*   **Webpage Generation:**  Generate web pages based on specific inputs.
*   **Trajectory Collection:**  Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides significant benefits to different user groups:

### For Agents Researchers & LLM Trainers

*   A **powerful baseline** superior to ReAct, serving as a strong starting point for model training and ablation studies.
*   **One-click evaluation scripts** streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:**  Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:**  Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A rich toolset and visual tracing tools make development and debugging intuitive.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools for an agent to use.
*   **Environment:** The context in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** A standardized workflow for evaluating a specific dataset, including preprocessing, rollout, and judging logic.

For more details on design and implementation, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent includes code and examples to help you get started quickly. Follow these steps to run your first agent, or use [`docker/README.md`](./docker/README.md) for a streamlined Docker setup with an interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. Use [uv](https://github.com/astral-sh/uv) for dependency management.

First, make sure Python and uv are installed.

Then clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

After copying the `.env.example` file, fill in the necessary keys in the `.env` file, e.g. LLM API keys. For example:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (September 1 – October 31, 2025). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Once you’ve applied, replace the API key in the .env file below:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with an interactive frontend.

### Quick Start

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

Launch an interactive CLI chatbot with this agent:

```bash
# NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --config simple/base_search
# To avoid using the search toolkit, run:
python scripts/cli_chat.py --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

The repository provides ready-to-use examples. Some examples require internet search capabilities, so you'll need to configure the tool APIs in the `.env` file under the tools module:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

For example, to enable the agent to search the web and generate an SVG image on "DeepSeek V3.1 New Features," run:

```bash
python examples/svg_generator/main.py
```

To visualize the agent’s runtime with a web UI, download the frontend package from the Youtu-Agent releases and install it:

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

Access the project via the local link shown in the terminal:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

Given a research topic, the agent automatically searches the web, collects information, and outputs an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking on standard datasets. For example, to evaluate on `WebWalkerQA`:

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

## 📖 Dive Deeper

Learn more about the framework and its capabilities:

-   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
-   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide to get you up and running.
-   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the excellent work of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions! See our [**Contributing Guidelines**](./CONTRIBUTING.md).

## 📚 Citation

If you find this work useful, cite:

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