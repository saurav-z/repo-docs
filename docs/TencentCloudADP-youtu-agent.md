# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent empowers you to create and deploy cutting-edge AI agents for diverse applications, leveraging the power of open-source models.  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent).

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

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent is a flexible, high-performance framework for building, running, and evaluating autonomous AI agents. This framework empowers you with advanced agent capabilities like data analysis, file processing, and in-depth research, all while utilizing open-source models.

**Key Features:**

*   **Exceptional Performance:** Achieved impressive results on WebWalkerQA and GAIA benchmarks using only DeepSeek-V3 models, showcasing a strong open-source foundation.
*   **Cost-Effective & Open-Source Focused:** Designed for accessible and budget-friendly deployment, avoiding reliance on expensive closed-source models.
*   **Practical Use Cases:** Supports common tasks such as CSV analysis, literature reviews, personal file organization, and upcoming podcast and video generation.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python) with extensibility for various model APIs (DeepSeek to gpt-oss), tool integrations, and framework implementations.
*   **Streamlined Automation:** Simplifies development with YAML-based configurations, automated agent generation, and easy setup.

## 🗞️ News

*   📺 **[2025-09-09]** Design Philosophy and Basic Usage Live Sharing: [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 **[2025-09-02]** Free DeepSeek API Tokens from Tencent Cloud International: New users of the DeepSeek API receive **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381). For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]** DeepSeek-V3.1 Update and Usage Live Sharing: [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and lightweight tools to deliver strong results in deep search and tool use benchmarks.

*   **WebWalkerQA:** Achieved **71.47%** accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art.
*   **GAIA:** Achieved **72.8%** pass@1 on the text-only validation subset, using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images to view detailed videos.

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
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Wide Research</strong><br>Generates a comprehensive report, replicating the functionality of Manus.
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
> Explore more examples and details in the [`examples`](./examples) directory and the [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automatic Agent Generation

Youtu-Agent significantly simplifies agent creation with **automatic agent configuration generation**.  Instead of writing code, create agents effortlessly with simple YAML-based configs.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Define your requirements interactively and automatically generate, then run agent configurations.
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
> Find more details in the [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keep the framework simple and easy to use.
*   **Modular & Configurable:** Flexible customization and easy integration of new components.
*   **Open-Source & Low-Cost:** Promote accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:**  In-depth analysis of tool calls and agent trajectories (coming soon).

### Automation

*   **YAML-Based Configuration:** Structured and manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generate configurations based on user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep / Wide Research:** Search-oriented tasks.
*   **Webpage Generation:** Generate web pages based on specific inputs.
*   **Trajectory Collection:** Collect data for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant benefits to different user groups:

### For Agents Researchers & LLM Trainers

*   **Simple Baseline:** A powerful starting point for model training and ablation studies.
*   **One-Click Evaluation Scripts:** Streamline the experimental process for consistent benchmarking.

### For Agent Application Developers

*   **Proven Scaffolding:** Build real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and a rich toolkit.
*   **Modular Design:** Customizable key components (e.g., `Environment`, `ContextManager`).

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Explore deep research, data analysis, and file organization examples.
*   **Simplicity & Debuggability:**  Intuitive development and debugging tools.

## 🧩 Core Concepts

*   **Agent:** LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can use.
*   **Environment:** Where the agent operates (e.g., a browser, shell).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Encapsulated workflow for specific datasets.

For more information, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to start using Youtu-Agent.  You can also use a streamlined Docker-based setup.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. Use [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure your API keys.
```

3.  Configure the `.env` file with necessary API keys (LLM, etc.).

```bash
# Example LLM configuration (DeepSeek)
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381).  After applying, update the `.env` file:

```bash
# LLM Configuration (Tencent Cloud DeepSeek)
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for Docker-based setup.

### Quick Start

Youtu-Agent includes built-in configurations. The `configs/agents/simple/base_search.yaml` file defines a simple agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run an interactive CLI chatbot:

```bash
#  Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --config simple/base
```

📖  [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in the `.env` file for examples using web search:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image on the topic of “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime status, install the frontend package:

```bash
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run:

```bash
python examples/svg_generator/main_web.py
```

Access the project at: `http://127.0.0.1:8848/`

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖  [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on standard datasets, such as `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖  [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get up and running quickly.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

See our [**Contributing Guidelines**](./CONTRIBUTING.md) for information on how to contribute.

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