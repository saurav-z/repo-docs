# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

> **Youtu-Agent is a flexible and high-performing agent framework, enabling you to build, run, and evaluate autonomous agents using accessible, open-source models. ([See the original repo](https://github.com/TencentCloudADP/youtu-agent))**

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
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

Youtu-Agent empowers developers to create sophisticated autonomous agents for tasks such as data analysis, file processing, and in-depth research, all while leveraging the power of open-source LLMs. It offers a robust, flexible, and cost-effective solution for building and deploying intelligent agents.

**Key Features:**

*   **Superior Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8% on text-only subset) using open-source models like DeepSeek-V3.
*   **Cost-Effective & Open-Source Focused:** Designed for efficient, low-cost deployment, eliminating reliance on expensive closed-source models.
*   **Practical Use Cases:** Includes out-of-the-box support for CSV analysis, research, file organization, and more, with podcast and video generation coming soon.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting diverse LLM APIs (DeepSeek, GPT-OSS), versatile tool integrations, and extensive framework implementations.
*   **Simplified Automation:** YAML-based configuration, automatic agent generation, and streamlined setup reduces development time and manual effort.

## 🗞️ News

*   **(2025-09-09)**: Live sharing the design philosophy and basic usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **(2025-09-02)**: [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **(2025-08-28)**: Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent excels on challenging benchmarks, demonstrating its capabilities using open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved **71.47%** accuracy with `DeepSeek-V3.1`, setting a new SOTA performance.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved **72.8% pass@1** on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324` (including models used within tools). Multimodal evaluation is in progress.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore practical applications with the following examples. Click the images to view videos demonstrating each use case.

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
      <strong>Wide Research</strong><br>Generates a comprehensive report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses and analyzes papers, compiling related literature.
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
>  Explore more details and examples in the [`examples`](./examples) directory and in the [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with **automatic agent configuration generation**. Unlike other frameworks, you can define agents using simple YAML-based configs. A built-in "meta-agent" interacts with you to capture requirements and automatically generates and saves the configuration.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define requirements, generate agent configurations, and run agents seamlessly.
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
>  Find out more details in the [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
*   **Minimal Design**: Keep the framework simple and easy to use, avoiding unnecessary overhead.
*   **Modular & Configurable**: Enable flexible customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost**: Promote accessibility and cost-effectiveness for various applications.

### Core Features
*   **Built on openai-agents**: Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
*   **Tracing & Analysis System**: Beyond OTEL, our `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
*   **YAML Based Configuration**: Structured and easily manageable agent configurations.
*   **Automatic Agent Generation**: Based on user requirements, agent configurations can be automatically generated.
*   **Tool Generation & Optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases
*   **Deep / Wide Research**: Covers common search-oriented tasks.
*   **Webpage Generation**: Examples include generating web pages based on specific inputs.
*   **Trajectory Collection**: Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed to provide significant value for various user groups:

### For Agents Researchers & LLM Trainers
*   A **simple yet powerful baseline** that is stronger than basic ReAct, serving as an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers
*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts
*   **Practical Use Cases**: The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability**: A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## 🧩 Core Concepts

*   **Agent**: An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit**: An encapsulated set of tools that an agent can use.
*   **Environment**: The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager**: A configurable module for managing the agent's context window.
*   **Benchmark**: An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For in-depth details on design and implementation, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent offers comprehensive code and examples to jumpstart your projects. Follow the steps below to launch your first agent, or see [`docker/README.md`](./docker/README.md) for a streamlined, Docker-based setup with an interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> This project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

Ensure Python and uv are installed.

Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

Configure the `.env` file with your necessary API keys, such as LLM API keys:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace the API key in the .env file with your key once you've applied:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

For a Docker-based setup, consult [`docker/README.md`](./docker/README.md).

### Quick Start

Youtu-Agent provides pre-built configurations. For example, the `configs/agents/simple/base_search.yaml` defines an agent equipped with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run an interactive CLI chatbot with the agent by executing:

```bash
# NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 Additional details in the [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart).

### Explore More Examples

The repository contains several ready-to-use examples. Configure tool APIs in the `.env` file under the tools module:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To have the agent search the web and generate an SVG image on "DeepSeek V3.1 New Features," run:

```bash
python examples/svg_generator/main.py
```

For web UI visualization, download the frontend package from the Youtu-Agent releases and install:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run the web version of the SVG image generation:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link when the terminal displays:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will gather info and output an SVG visualization given a research topic.

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

View and analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

After getting started, access comprehensive documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Answers to common questions.

## 🙏 Acknowledgements

This project utilizes the work of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome community contributions. See our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

## 📚 Citation

If you use this work, please cite it:

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