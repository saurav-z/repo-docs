# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Unlock the power of autonomous agents with Youtu-Agent, a flexible and high-performing framework for building, running, and evaluating AI agents.** ([See the original repository](https://github.com/Tencent/Youtu-agent))

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<!-- <a href=https://arxiv.org/abs/2502.14345><img src=https://img.shields.io/badge/arXiv-2502.14345-b31b1b.svg></a> -->
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

Youtu-Agent empowers you to create intelligent agents for tasks like data analysis, file processing, and research, all while leveraging the power of open-source models.

**Key Features:**

*   **High Performance:** Achieves impressive results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%), demonstrating strong capabilities with open-source models like DeepSeek-V3.
*   **Open-Source and Cost-Effective:** Optimized for accessible deployment without relying on expensive closed-source models.
*   **Practical Use Cases:** Includes out-of-the-box support for CSV analysis, literature review, file organization, and more.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python) and supports diverse model APIs and tool integrations.
*   **Automated Workflow:** Simplify development with YAML-based configurations, automatic agent generation, and streamlined setup.

## 📢 News

*   **[2025-09-02]**:  Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`!  For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]**:  Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework.  [Documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels on deep search and tool-use benchmarks, showcasing its strength with open-source models.

*   **WebWalkerQA**:  Achieved 71.47% accuracy with DeepSeek-V3.1, representing a new state-of-the-art.
*   **GAIA**:  Reached 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore Youtu-Agent's capabilities through these interactive examples. Click on the images to view detailed videos.

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

Youtu-Agent simplifies agent creation with automated configuration generation.

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

Refer to the [`examples`](./examples) directory and  [`docs/examples.md`](./docs/examples.md) for more advanced use-cases.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Simple and easy to use, avoiding unnecessary overhead.
*   **Modular & Configurable:** Easy integration of new components.
*   **Open-Source Model Support & Low-Cost:**  Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:**  Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation

*   **YAML Based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:**  Agent configurations can be automatically generated.
*   **Tool Generation & Optimization:**  Tool evaluation and automated optimization will be supported in the future.

### Use Cases

*   **Deep / Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:** Generating web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent delivers value for:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** that is stronger than basic ReAct.
*   **One-click evaluation scripts**.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases**: The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability**:  Intuitive development and debugging.

## 🧩 Core Concepts

*   **Agent**: An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit**:  An encapsulated set of tools.
*   **Environment**: The world in which the agent operates.
*   **ContextManager**: A configurable module for managing the agent's context window.
*   **Benchmark**: An encapsulated workflow.

Refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for details.

## 🚀 Getting Started

Follow the steps below to run your first agent, or refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Setup

#### Source Code Deployment

> \[!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

Fill in the necessary keys in the `.env` file (e.g., LLM API keys).

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

Or, try the free DeepSeek API tokens from Tencent Cloud:

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

Run an interactive CLI chatbot with a search tool:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in the `.env` file:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

For the web UI (install frontend package first):

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl

python examples/svg_generator/main_web.py
```

Then access the project:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

To evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

See [Evaluation Analysis](./frontend/exp_analysis/README.md) for result analysis.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project uses these open-source projects:

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
  howpublished = {\url{https://github.com/TencentCloudADP/youtu-agent}},
}
```

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)