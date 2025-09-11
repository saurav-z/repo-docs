# 🤖 Youtu-Agent: Build Powerful Agents with Open-Source Models

**Unleash the power of open-source models to create advanced autonomous agents with Youtu-Agent, a flexible and high-performance framework.** [Explore the Youtu-Agent Repository](https://github.com/Tencent/Youtu-agent)

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a> | <a href="README_JA.md"><b>日本語</b></a> | <a href="#-benchmark-performance"><b>🌟 Performance</b></a> | <a href="#-examples"><b>💡 Examples</b> </a> | <a href="#-features"><b>✨ Features</b> </a> | <a href="#-getting-started"><b>🚀 Getting Started</b> </a> | <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent is a robust and versatile framework designed for building, running, and evaluating autonomous agents, delivering impressive results with open-source models. This framework excels in diverse applications, including data analysis, file processing, and in-depth research.

**Key Features:**

*   ✅ **Exceptional Performance**: Achieved impressive results on WebWalkerQA and GAIA benchmarks using open-source DeepSeek-V3 models.
*   💰 **Cost-Effective & Open-Source Friendly**: Optimized for accessible, low-cost deployment without relying on proprietary models.
*   🛠️ **Practical Use Cases**: Includes out-of-the-box support for CSV analysis, literature reviews, file organization, and more.
*   ⚙️ **Flexible Architecture**: Built upon the [openai-agents](https://github.com/openai/openai-agents-python) foundation, supporting diverse model APIs and tool integrations.
*   🤖 **Automation & Simplicity**: Streamlined agent creation via YAML-based configurations and automatic agent generation, reducing manual setup.

## 🗞️ News

*   🎁 **[2025-09-02]**:  Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381). For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]**: Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [Documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent's performance shines on challenging benchmarks, demonstrating strong capabilities with open-source models and lightweight tools.

*   **WebWalkerQA**:  Achieved 71.47% accuracy with DeepSeek-V3.1, establishing a new state-of-the-art performance.
*   **GAIA**: Achieved 72.8% pass@1 on the text-only validation subset, showcasing its effectiveness in complex reasoning tasks.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore Youtu-Agent's capabilities through these interactive examples. Click on the images to view detailed videos.

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyze CSV files and generate HTML reports.
      <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920"
             poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Organize and categorize local files.
      <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e"
             poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Wide Research</strong><br>Generate comprehensive reports.
       <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92"
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height=300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyze papers and compile related literature.
      <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d"
             poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg"
             controls muted preload="metadata"
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its ability to automatically generate agent configurations.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively generate and run agent configurations.
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

Explore detailed examples and advanced use cases in the [`examples`](./examples) directory and documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   💡 **Minimal Design**: Focus on simplicity and ease of use.
*   ⚙️ **Modular & Configurable**: Enables flexible customization and integration.
*   💸 **Open-Source Model Support & Low-Cost**: Promotes accessibility and cost-effectiveness.

### Core Features

*   🚀 **Built on openai-agents**: Leverage openai-agents' streaming, tracing, and agent-loop capabilities.
*   ⚡ **Fully Asynchronous**: Ensures high-performance execution.
*   🔍 **Tracing & Analysis System**:  In-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation

*   📄 **YAML Based Configuration**:  Structured and manageable agent configurations.
*   🤖 **Automatic Agent Generation**: Generates agent configurations based on user requirements.
*   🛠️ **Tool Generation & Optimization**:  Automated tool evaluation, optimization, and customized tool generation planned.

### Use Cases

*   📚 Deep / Wide Research
*   🕸️ Webpage Generation
*   📊 Trajectory Collection

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides significant benefits for diverse user groups:

### For Agents Researchers & LLM Trainers

*   ✅ A **simple yet powerful baseline** for model training and ablation studies.
*   ⚙️  **One-click evaluation scripts** for streamlined benchmarking.

### For Agent Application Developers

*   🏗️  A **proven and portable scaffolding** for building real-world agent applications.
*   🎯  **Ease of Use**: Get started quickly with simple scripts and toolkits.
*   🧩  **Modular Design**: Key components are encapsulated and highly customizable.

### For AI & Agent Enthusiasts

*   🚀 **Practical Use Cases**:  Examples in the `/examples` directory cover diverse tasks.
*   💡 **Simplicity & Debuggability**:  Intuitive development and debugging through a rich toolset.

## 🧩 Core Concepts

*   **Agent**: LLM configured with prompts, tools, and an environment.
*   **Toolkit**: Encapsulated tools for agent use.
*   **Environment**: Where the agent operates.
*   **ContextManager**: Manages the agent's context window.
*   **Benchmark**: Workflow for dataset evaluation.

For more detailed design and implementation information, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Quickly get started with Youtu-Agent by following these steps, or use the Docker setup in [`docker/README.md`](./docker/README.md).

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys in .env
```

3.  Configure necessary API keys in the `.env` file, such as your LLM API keys.

    *   DeepSeek API example:

```bash
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

    *   Tencent Cloud International DeepSeek API example (free tokens available):

```bash
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for Docker setup.

### Quick Start

Use the default configuration to launch an interactive CLI chatbot:

```bash
#  Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
python scripts/cli_chat.py --stream --config default
# To run without search:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs (e.g., `SERPER_API_KEY`, `JINA_API_KEY`) in `.env` for examples requiring internet access.

To generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

For the web UI, download and install the frontend package:

```bash
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.6/utu_agent_ui-0.1.6-py3-none-any.whl
uv pip install utu_agent_ui-0.1.6-py3-none-any.whl
```

Run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project through the local link after the server starts.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will search, collect information, and output an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on datasets like `WebWalkerQA`:

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` and your `exp_id`. Use `WebWalkerQA_15` for a quick eval.
# Configure `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` in `.env`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Explore the framework with our comprehensive documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide to get you up and running.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions and issues.

## 🙏 Acknowledgements

This project builds upon the excellent work of several open-source projects:
* [openai-agents](https://github.com/openai/openai-agents-python)
* [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
* [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome community contributions! See our [**Contributing Guidelines**](./CONTRIBUTING.md).

## 📚 Citation

If you find this work useful, cite us:

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