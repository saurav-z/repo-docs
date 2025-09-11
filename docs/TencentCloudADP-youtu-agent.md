# 🚀 Youtu-Agent: Build Powerful Agents with Open-Source Models

Youtu-Agent empowers you to build high-performing autonomous agents using open-source models, enabling data analysis, file processing, and in-depth research.  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent).

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

**Key Features:**

*   ✅ **High-Performance with Open-Source Models:** Achieved SOTA results on WebWalkerQA (71.47%) and GAIA (72.8%), demonstrating strong performance with DeepSeek-V3 series models, without relying on closed-source models.
*   💰 **Cost-Effective & Open-Source Focused:** Designed for accessible, low-cost deployment, reducing dependence on proprietary models.
*   💼 **Practical Use Cases:** Supports CSV analysis, literature review, file organization, and upcoming podcast/video generation.
*   🧩 **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), with support for various model APIs (e.g., DeepSeek, gpt-oss), tool integrations, and framework implementations.
*   ⚙️ **Simplified Development:** YAML-based configurations, automated agent generation, and streamlined setup minimize manual effort.

## 🗞️ News

*   🎁 **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]** Live sharing about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. Documentation: [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and lightweight tools to achieve impressive results on challenging benchmarks.

*   **WebWalkerQA:** Achieved **71.47% accuracy** with DeepSeek-V3.1, setting a new SOTA performance.
*   **GAIA:** Achieved **72.8% pass@1** on the text-only validation subset using DeepSeek-V3-0324.  Multimodal tool evaluation is in progress.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click on the images to view detailed videos.

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

Youtu-Agent features **automatic agent configuration generation**, simplifying the creation of task-specific agents. Instead of complex coding or prompt engineering, use YAML-based configurations for streamlined automation. A "meta-agent" guides you through requirements, then generates and saves the configuration.

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

For more examples and advanced use-cases, explore the [`examples`](./examples) directory and the documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeps the framework simple and easy to use.
*   **Modular & Configurable:** Supports flexible customization and integration of new components.
*   **Open-Source & Low-Cost:** Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Inherits the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories. (Coming soon).

### Automation

*   **YAML-based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Agent configurations can be automatically generated based on user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep / Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:**  Examples include generating web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant advantages for various user groups:

### For Agents Researchers & LLM Trainers

*   **Simple yet Powerful Baseline:** Serves as an excellent starting point for model training and ablation studies.
*   **One-Click Evaluation Scripts:** Streamlines the experimental process and ensures consistent benchmarking.

### For Agent Application Developers

*   **Proven and Portable Scaffolding:**  For building real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and built-in toolkits.
*   **Modular Design:** Encapsulated and customizable key components (e.g., `Environment`, `ContextManager`).

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:**  Rich toolset and visual tracing tools for intuitive development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools for an agent.
*   **Environment:** Where the agent operates (e.g., browser, shell).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:**  An encapsulated workflow for a specific dataset.

For more details, refer to the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to run your first agent or use the Docker setup:

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. Recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  #  Configure API keys.
```

Populate `.env` with necessary keys, such as:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free.

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

See [`docker/README.md`](./docker/README.md) for a Docker-based setup with a UI.

### Quick Start

Run a CLI chatbot with the default agent:

```bash
# NOTE: Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` for examples needing web search:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate SVG image on “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

For the web UI, download and install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.6/utu_agent_ui-0.1.6-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.6-py3-none-any.whl
```

Run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project at the local link provided.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on WebWalkerQA:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results can be analyzed in the evaluation platform.  See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Explore the framework with comprehensive documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Step-by-step guide to get started.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Answers to common questions.

## 🙏 Acknowledgements

This project is built upon:
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Contribute to Youtu-Agent!  See our [**Contributing Guidelines**](./CONTRIBUTING.md).

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