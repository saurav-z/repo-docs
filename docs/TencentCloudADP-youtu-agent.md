# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a versatile framework that empowers you to create, run, and evaluate autonomous agents, achieving impressive results using open-source models.  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent)

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

[![Youtu-agent Logo](docs/assets/mascot.png)](https://github.com/TencentCloudADP/youtu-agent)

**Key Features:**

*   ✅ **High Performance:** Achieves strong results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%) using open-source models (DeepSeek-V3).
*   💡 **Open-Source & Cost-Effective:** Designed for accessibility and efficient deployment with open-source models, minimizing costs.
*   ⚙️ **Practical Use Cases:** Includes ready-to-use agents for CSV analysis, literature review, file organization, and more.
*   🧩 **Flexible Architecture:** Built on the OpenAI Agents framework, supporting various model APIs (DeepSeek, GPT-OSS) and tool integrations.
*   🤖 **Automated Agent Generation:** Streamlines agent creation using YAML-based configurations and an auto-generation feature, saving time and effort.

## ✨ Core Features

*   **Based on openai-agents:** Our framework builds on the openai-agents SDK, ensuring compatibility with `responses` and `chat.completions` APIs for model flexibility.
*   **Fully asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & analysis system:** A `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories (coming soon).

### Automation
- **YAML based configuration:** Structured and easily manageable agent configurations.
- **Automatic agent generation:** Agent configurations can be automatically generated based on user requirements.
- **Tool generation & optimization:** Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Design Philosophy
- **Minimal design:** Keeping the framework simple and easy to use.
- **Modular & configurable:** Flexible customization and easy integration of new components.
- **Open-source model support & low-cost:** Promotes accessibility and cost-effectiveness for various applications.

### Use Cases
- **Deep / Wide research**: Covers common search-oriented tasks.
- **Webpage generation**: Examples include generating web pages based on specific inputs.
- **Trajectory collection**: Supports data collection for training and research purposes.

## 🌟 Benchmark Performance

Youtu-Agent is built on open-source models and lightweight tools, demonstrating strong results on challenging deep search and tool use benchmarks.

*   **WebWalkerQA:** Achieved 71.47% accuracy with `DeepSeek-V3.1`.
*   **GAIA:** Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

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

A key feature is its ability to automatically generate agent configurations using YAML-based configs: a built-in "meta-agent" chats with you to capture requirements, then generates and saves the config automatically.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
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

## 🤔 Why Choose Youtu-Agent?

`Youtu-Agent` is designed to provide significant value to different user groups:

### For Agents Researchers & LLM Trainers
- A **simple yet powerful baseline** that is stronger than basic ReAct, serving as an excellent starting point for model training and ablation studies.
- **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers
- A **proven and portable scaffolding** for building real-world agent applications.
- **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
- **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts
- **Practical Use Cases**: The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
- **Simplicity & Debuggability**: A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## 🚀 Getting Started

Follow these steps to get started with Youtu-Agent:

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

Fill in necessary keys in the `.env` file, e.g. LLM API keys.  For example:

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

### Quick Start

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

```bash
# For example, to enable the agent to automatically search online for information and generate an SVG image on the topic of “DeepSeek V3.1 New Features,” run the following command:
python examples/svg_generator/main.py
```

If you want to visualize the agent’s runtime status using the web UI:

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

### Run Evaluations

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

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide to get you up and running.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions and issues.

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Read our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

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