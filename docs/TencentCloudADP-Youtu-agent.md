<!--
  _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( Y | o | u | t | u | - | A | g | e | n | t | : | P | o | w | e | r | i | n | g | A | u | t | o )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
-->

# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Empower your AI projects with Youtu-Agent, a high-performance agent framework delivering advanced capabilities using open-source models. [Explore the Youtu-Agent repository](https://github.com/TencentCloudADP/Youtu-agent).**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#examples"><b>💡 Examples</b></a>
| <a href="#features"><b>✨ Features</b></a>
| <a href="#getting-started"><b>🚀 Getting Started</b></a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b></a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="right" style="margin-left:20px;">

Youtu-Agent is a flexible and high-performance framework designed for building, running, and evaluating autonomous agents, offering strong performance with open-source models. It provides a cost-effective solution for diverse agent applications.

**Key Features:**

*   **High Performance with Open Source:** Achieve state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%), all using open-source models (DeepSeek-V3 series).
*   **Cost-Effective & Accessible:** Designed for deployment with open-source models, reducing reliance on expensive closed models and ensuring accessibility.
*   **Practical Use Cases:** Supports tasks like data analysis, file management, and research, with more coming soon, including podcast and video generation.
*   **Flexible & Extensible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), it supports a range of model APIs (DeepSeek, gpt-oss), and tool integrations.
*   **Simplified Development:** YAML-based configuration, automatic agent generation, and streamlined setup reduce manual coding and complexity.

## What's New

*   **[2025-09-09]**: Design philosophy and basic usage sharing. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **[2025-09-02]**: DeepSeek API users get **3 million free tokens** (**Sep 1 – Oct 31, 2025**) from [Tencent Cloud International](https://www.tencentcloud.com/). Try it out for free! For enterprise solutions, see [Agent Development Platform](https://adp.tencentcloud.com).
*   **[2025-08-28]**: Updates on DeepSeek-V3.1 and its integration with the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## Benchmark Performance

Youtu-Agent demonstrates strong performance on challenging benchmarks using open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 71.47% accuracy with `DeepSeek-V3.1`.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## Examples

Explore various agent capabilities through these examples:

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files.
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
      <strong>Wide Research</strong><br>Gathers extensive information for a comprehensive report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyzes a paper and compiles related literature.
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
> Find detailed examples in the [`examples`](./examples) directory and the [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation using YAML files.

```bash
# Interactively clarify requirements, and automatically generate the config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively clarifies your requirements, automatically generates the agent configuration, and runs it.
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
> More details available in the [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/).

## Features

### Design Philosophy

*   **Minimal Design:** Simple, easy-to-use framework.
*   **Modular & Configurable:** Highly customizable with easy component integration.
*   **Open-Source & Low-Cost:** Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on OpenAI Agents:** Leverages [openai-agents](https://github.com/openai/openai-agents-python) for streaming, tracing, and agent-loop capabilities.
*   **Fully Asynchronous:** High-performance and efficient execution.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories.

### Automation

*   **YAML-based Configuration:** Structured and easily managed agent configurations.
*   **Automatic Agent Generation:** Automatically generated agent configurations based on user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, automated optimization, and custom tool generation.

### Use Cases

*   Deep / Wide Research
*   Webpage Generation
*   Trajectory Collection

## Why Choose Youtu-Agent?

Youtu-Agent provides value for various user groups:

### For Agents Researchers & LLM Trainers

*   A powerful baseline for model training and ablation studies.
*   One-click evaluation scripts for streamlined experimentation.

### For Agent Application Developers

*   A proven and portable scaffolding for building agent applications.
*   Ease of Use: Simplified scripts and a rich set of built-in toolkits.
*   Modular Design: Encapsulated and customizable core components.

### For AI & Agent Enthusiasts

*   Practical use cases in the `/examples` directory.
*   Simplicity & Debuggability: Rich toolset and visual tracing tools for development.

## Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools.
*   **Environment:** The operating environment (e.g., browser, shell).
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for datasets, including preprocessing, rollout, and judging logic.

Find more information in the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## Getting Started

Follow these steps to start using Youtu-Agent:

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys.
```

Populate the `.env` file with the necessary API keys, such as LLM API keys.

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace API key in the .env file below after application.

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Quick Start

Run a simple agent with a search tool:

```bash
# Configure `SERPER_API_KEY` and `JINA_API_KEY` in `.env`.
python scripts/cli_chat.py --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Enable internet search tools in the `.env` file.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image about "DeepSeek V3.1 New Features".

```bash
python examples/svg_generator/main.py
```

For the web UI, download the frontend package from the Youtu-Agent releases.

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the displayed local link.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on datasets like `WebWalkerQA`.

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze the results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Detailed guide to get you started.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Answers to common questions.

## Acknowledgements

This project is built upon:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## Contributing

Read the [**Contributing Guidelines**](./CONTRIBUTING.md) if you'd like to help improve Youtu-Agent.

## Citation

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

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)