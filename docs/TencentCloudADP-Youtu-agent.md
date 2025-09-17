# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent is a flexible and high-performing agent framework that empowers you to build and deploy AI agents using open-source models. [Explore the project on GitHub](https://github.com/TencentCloudADP/Youtu-agent)!**

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

Youtu-Agent streamlines agent development with features like automatic configuration generation, versatile use cases, and strong performance.

**Key Features:**

*   ✅ **High-Performance:** Achieves impressive benchmark results, demonstrating strong capabilities with open-source models like DeepSeek-V3.
    *   **WebWalkerQA:** 71.47% (pass@1)
    *   **GAIA (text-only subset):** 72.8% (pass@1)
*   💰 **Cost-Effective:** Optimized for open-source models, reducing reliance on expensive closed-source alternatives.
*   🛠️ **Practical Use Cases:** Supports data analysis, file processing, and more.
*   ⚙️ **Flexible Architecture:** Built on openai-agents, extensible for various model APIs and tool integrations.
*   🤖 **Automated Agent Creation:** YAML-based configurations and automatic generation simplify setup.

## 🗞️ News

*   📺 [2025-09-09] Live sharing design philosophy and usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 [2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 [2025-08-28] Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent delivers strong results on challenging benchmarks, leveraging open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA):** 71.47% accuracy with DeepSeek-V3.1.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/):** 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) with DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore practical applications with these examples:

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
      <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report, replicating the functionality of Manus.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a given paper, performs analysis, and compiles related literature.
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

> [!NOTE]
> See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automated configuration generation.

```bash
# Interactively specify requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Clarify requirements and instantly generate, and run an agent configuration.
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

*   **Minimal Design:** Keeps the framework simple and user-friendly.
*   **Modular & Configurable:** Offers flexible customization and easy component integration.
*   **Open-Source Focus:** Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leveraging openai-agents SDK.
*   **Fully Asynchronous:** Enables high-performance and efficient execution, especially for benchmarking.
*   **Tracing & Analysis:** In-depth analysis of tool calls and agent trajectories.

### Automation

*   **YAML-based Configuration:** Structured and manageable agent configurations.
*   **Automatic Agent Generation:** Based on user requirements, agent configurations can be automatically generated.
*   **Tool generation & optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:** Examples include generating web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent caters to various user groups:

### For Agents Researchers & LLM Trainers

*   **Strong Baseline:** Serves as a solid starting point for model training and ablation studies.
*   **One-click evaluation:** Streamlines experimentation with scripts for benchmarking.

### For Agent Application Developers

*   **Portable Scaffolding:** A proven framework for real-world agent applications.
*   **Ease of Use:** Quick start and built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Example tasks like report generation and data analysis.
*   **Simplicity & Debuggability:** Rich toolset and visual tracing tools for development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can use.
*   **Environment:** The context in which the agent operates.
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Workflow for evaluating performance on a dataset.

Refer to the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for more details.

## 🚀 Getting Started

Follow these steps to get started:

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and uv for dependency management.

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys.
```

Fill in the `.env` file with your API keys (e.g., LLM API keys).

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Quick Start

Use the default config (`configs/agents/default.yaml`) for a simple agent with a search tool.

```bash
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --stream --config default
# For the base config
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` for internet search capabilities.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image on “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

Use the web UI for visualization:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project at `http://127.0.0.1:8848/`.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Benchmark on standard datasets. Example for `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and analyzed in the evaluation platform.

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

Read the [**Contributing Guidelines**](./CONTRIBUTING.md) for help.

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