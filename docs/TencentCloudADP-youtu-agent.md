# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent empowers you to build, run, and evaluate AI agents efficiently using open-source models.  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent)

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#-key-features"><b>✨ Key Features</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#-examples"><b>💡 Examples</b> </a>
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent is a flexible and high-performance framework for building, running, and evaluating autonomous agents, offering powerful capabilities with open-source models.  It goes beyond basic benchmarks to offer data analysis, file processing, and deep research, all while being cost-effective and open-source friendly.

## ✨ Key Features

*   **High Performance with Open-Source:** Achieved impressive results, including 71.47% on WebWalkerQA (pass@1) and 72.8% on GAIA (text-only subset, pass@1), using DeepSeek-V3 series models.
*   **Cost-Effective and Open-Source Focused:** Optimized for accessible, low-cost deployments without reliance on proprietary models, enabling wider adoption.
*   **Practical Use Cases Out-of-the-Box:**  Includes ready-to-use tools for tasks like CSV analysis, literature reviews, personal file organization, and video/podcast generation (coming soon).
*   **Flexible and Extensible Architecture:** Built on the [openai-agents](https://github.com/openai/openai-agents-python) framework, supporting various model APIs (DeepSeek, gpt-oss), tool integrations, and framework implementations.
*   **Simplified Automation:** YAML-based configurations, automatic agent generation, and streamlined setup minimize manual effort.

## 🗞️ News

*   **2025-09-09:**  Live sharing of design philosophy and basic usage. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **2025-09-02:** [Tencent Cloud International](https://www.tencentcloud.com/) offers new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**). Try it out and explore enterprise agent solutions with the [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **2025-08-28:** Live sharing of DeepSeek-V3.1 updates and usage. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and lightweight tools to deliver outstanding results on challenging deep search and tool use benchmarks.

*   **WebWalkerQA:** Achieved 71.47% accuracy with `DeepSeek-V3.1`, surpassing previous SOTA results.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.  Multimodal tool evaluation is actively being extended.

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
> Explore the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its **automatic agent generation** feature.

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

> [!NOTE]
> See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal design:**  Keep the framework simple and easy to use.
*   **Modular & configurable:**  Facilitate flexible customization and integration.
*   **Open-source model support & low-cost:** Promote accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leveraging the openai-agents SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
*   **Tracing & analysis system**:  Our `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation

*   **YAML based configuration**: Streamline agent management.
*   **Automatic agent generation**:  Generate configurations based on user needs.
*   **Tool generation & optimization**: Future support for tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep / Wide research**:  Tackling common search-oriented tasks.
*   **Webpage generation**: Examples include generating web pages based on specific inputs.
*   **Trajectory collection**: Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers value for diverse user groups:

### For Agents Researchers & LLM Trainers

*   A **powerful baseline**, stronger than ReAct, for model training and research.
*   **One-click evaluation scripts** to streamline experimentation and benchmarking.

### For Agent Application Developers

*   A **proven scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and built-in toolkits.
*   **Modular Design**: Highly customizable core components such as `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases**: Examples in the `/examples` directory for deep research, data analysis, and more.
*   **Simplicity & Debuggability**: A rich toolset and visual tracing tools for intuitive development.

## 🧩 Core Concepts

*   **Agent**: An LLM configured with prompts, tools, and an environment.
*   **Toolkit**:  An encapsulated set of tools for agent use.
*   **Environment**:  The agent's operating context (e.g., browser, shell).
*   **ContextManager**:  A module for managing the agent's context window.
*   **Benchmark**: An encapsulated workflow for specific datasets, including preprocessing and evaluation.

For more detailed information, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to run your first agent. For a streamlined Docker setup, see [`docker/README.md`](./docker/README.md).

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. We recommend [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and `uv`.
2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure necessary API keys here!
```

Configure the `.env` file with your API keys, particularly LLM API keys (e.g., for DeepSeek or other compatible providers).

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for Docker setup instructions.

### Quick Start

Use a pre-configured agent with a search tool:

```bash
# Configure `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid search toolkit, run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` for examples requiring web search:

```bash
# Configure tool APIs
SERPER_API_KEY=<Get API Key>
JINA_API_KEY=<Get API Key>
```

Generate an SVG image on the topic of “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

For a web UI visualization, install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then, run the web version of the SVG generation example:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link shown in the terminal.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will automatically search the web and create an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate Youtu-Agent on standard datasets (e.g., WebWalkerQA):

```bash
# Prepare dataset (downloads and saves WebWalkerQA data)
python scripts/data/process_web_walker_qa.py

# Run evaluation with custom exp_id (WebWalkerQA_15 for quick tests)
# Ensure `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` are set in `.env`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

-   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the framework in detail.
-   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get started quickly.
-   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**:  Find answers to common questions.

## 🙏 Acknowledgements

We appreciate the contributions of these open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Contributions are welcome!  See our [**Contributing Guidelines**](./CONTRIBUTING.md).

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