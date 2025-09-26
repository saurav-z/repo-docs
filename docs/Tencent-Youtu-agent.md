# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent empowers developers to create sophisticated AI agents, offering state-of-the-art performance and cost-effective deployment using open-source models.** ([View on GitHub](https://github.com/Tencent/Youtu-agent))

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

Youtu-Agent is a flexible and high-performance framework designed for building, running, and evaluating autonomous agents.  It excels in tasks like data analysis, file processing, and deep research, leveraging open-source models for accessible and cost-effective deployments.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   **Exceptional Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%), utilizing open-source models like DeepSeek-V3 series.
*   **Open-Source & Cost-Effective:** Designed for easy and low-cost deployment, avoiding reliance on expensive, closed-source models.
*   **Practical Use Cases:** Supports a variety of applications, including CSV analysis, literature reviews, file organization, and more.
*   **Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), with extensive support for various model APIs, tool integrations, and framework implementations.
*   **Simplified Automation:**  Offers YAML-based configuration, automatic agent generation, and streamlined setup to reduce development overhead.

## 📰 News

*   **2025-09-09:**  Live sharing session on design philosophy and basic usage [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **2025-09-02:** Tencent Cloud International offers new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381)!  Also, explore the [Agent Development Platform](https://adp.tencentcloud.com) (ADP) for enterprise agent solutions.
*   **2025-08-28:** Live session covering DeepSeek-V3.1 integration in Youtu-Agent [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent showcases strong results on deep search and tool use benchmarks, powered by open-source models and lightweight tools.

*   **WebWalkerQA:**  Achieved **71.47%** accuracy using `DeepSeek-V3.1`, surpassing previous state-of-the-art.
*   **GAIA:**  Reached **72.8%** pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.  Multimodal tool evaluations are in development.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the capabilities of Youtu-Agent through these interactive examples. Click on the images to view detailed videos.

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
      <strong>Wide Research</strong><br>Generates a comprehensive report by gathering extensive information.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a paper, analyzes it, and compiles related literature.
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
> Explore the [`examples`](./examples) directory and the [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for detailed information.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its automatic agent generation feature. Define your agent's tasks with simple YAML configurations.  A built-in "meta-agent" assists in defining requirements and automatically generates the configuration.

```bash
# Interactively specify requirements and generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Simplify agent creation with interactive requirement capture and automatic configuration generation.
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
> Refer to the [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Focus on simplicity and ease of use, minimizing overhead.
*   **Modular & Configurable:** Flexible customization and straightforward integration of new components.
*   **Open-Source & Low-Cost:** Emphasizes accessibility and cost-effectiveness for diverse applications.

### Core Features

*   **Built on openai-agents:** Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python) for streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs, and supporting various models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution, particularly beneficial for benchmark evaluation.
*   **Tracing & Analysis System:** The `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories (coming soon).

### Automation

*   **YAML-Based Configuration:** Structured agent configurations that are easily managed.
*   **Automatic Agent Generation:** Generate agent configurations automatically based on user requirements.
*   **Tool Generation & Optimization:** Tool evaluation, automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide Research:**  Handles common search-oriented tasks.
*   **Webpage Generation:** Includes examples of generating web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed to be valuable for various user groups:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline**, surpassing basic ReAct, ideal for model training and ablation studies.
*   **One-click evaluation scripts** streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable foundation** for creating real-world agent applications.
*   **Ease of Use:** Quickly get started with straightforward scripts and a rich set of built-in toolkits.
*   **Modular Design:**  Encapsulated and highly customizable components such as `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** The `/examples` directory offers a range of tasks, including deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A comprehensive toolset and visual tracing tools enhance development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The context in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, encompassing preprocessing, rollout, and judging logic.

Find detailed design and implementation details in our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to quickly launch your first Youtu-Agent, or refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Ensure Python and uv are installed.
2.  Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure API keys after copying.
    ```

3.  Populate the `.env` file with the necessary API keys (e.g., LLM API keys).  For DeepSeek API users, you can obtain **3 million free tokens** (**Sep 1 – Oct 31, 2025**) from [Tencent Cloud International](https://www.tencentcloud.com/) after applying. Replace the example key in the `.env` file accordingly.

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Quick Start

Youtu-Agent includes example configurations. For instance, `configs/agents/simple/base_search.yaml` defines an agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run the interactive CLI chatbot with:

```bash
# Configure `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --config simple/base_search
# Run without search toolkit:
python scripts/cli_chat.py --config simple/base
```

📖 Learn more in the [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart).

### Explore More Examples

Configure tool APIs in the `.env` file for internet search-enabled agents:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To generate an SVG image based on "DeepSeek V3.1 New Features," run:

```bash
python examples/svg_generator/main.py
```

For the web UI, download and install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project through the local link displayed after the server starts.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

Given a topic, the agent searches, collects information, and generates an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Discover more in the [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples).

### Run Evaluations

To evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more in the [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval).

## 📖 Dive Deeper

Enhance your understanding of the framework with our comprehensive documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**:  A detailed guide for getting started.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**:  Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the excellent work of these open-source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions!  Review our [**Contributing Guidelines**](./CONTRIBUTING.md) to begin.

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