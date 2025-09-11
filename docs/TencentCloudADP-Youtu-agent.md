# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Unlock the potential of autonomous agents with **Youtu-Agent**, a versatile framework that delivers impressive results using open-source models. For the original repository, visit [TencentCloudADP/Youtu-agent](https://github.com/TencentCloudADP/Youtu-agent).

<div align="center">
  <a href="https://tencentcloudadp.github.io/youtu-agent/"><img src="https://img.shields.io/badge/📖-Documentation-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/TencentCloudADP/youtu-agent"><img src="https://img.shields.io/badge/GitHub-Tencent-blue.svg" alt="GitHub"></a>
  <a href="https://deepwiki.com/TencentCloudADP/youtu-agent"><img src="https://img.shields.io/badge/DeepWiki-Tencent-blue.svg" alt="DeepWiki"></a>
</div>

<p align="center">
  <a href="README_ZH.md"><b>中文</b></a> |
  <a href="README_JA.md"><b>日本語</b></a> |
  <a href="#-benchmark-performance"><b>🌟 Performance</b></a> |
  <a href="#-examples"><b>💡 Examples</b></a> |
  <a href="#-features"><b>✨ Features</b></a> |
  <a href="#-getting-started"><b>🚀 Getting Started</b></a> |
  <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b></a>
</p>

Youtu-Agent offers a flexible, high-performance solution for building, running, and evaluating AI agents, achieving top benchmark scores with open-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **Exceptional Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8% on the text-only subset) using DeepSeek-V3 series models.
*   💰 **Cost-Effective & Open-Source:** Designed for accessible and low-cost deployment, minimizing reliance on proprietary models.
*   🛠️ **Practical Use Cases:** Supports a wide array of tasks including CSV analysis, literature review, file organization, and more, with podcast and video generation coming soon.
*   ⚙️ **Flexible Architecture:** Built on the robust [openai-agents](https://github.com/openai/openai-agents-python) foundation, with extensive support for various model APIs (DeepSeek, gpt-oss, etc.), tool integrations, and framework implementations.
*   ⚙️ **Automation & Simplicity:** Simplifies agent creation with YAML-based configurations and automatic agent generation, reducing manual setup.

## 🗞️ News

*   🎁 \[2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 \[2025-08-28] We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent delivers strong results on challenging deep search and tool use benchmarks leveraging open-source models and lightweight tools.

*   **WebWalkerQA:** 71.47% accuracy with DeepSeek-V3.1, setting a new state-of-the-art.
*   **GAIA (Text-Only Subset):** 72.8% pass@1 using DeepSeek-V3-0324. Multimodal tool integration and full GAIA benchmark evaluation are underway.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore Youtu-Agent's capabilities through these interactive examples:

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
      <strong>Wide Research</strong><br>Generates a comprehensive report based on extensive information gathering.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a paper, performs analysis, and compiles related literature.
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

Youtu-Agent's **automatic agent generation** simplifies the agent creation process using YAML-based configurations.

```bash
# Interactively define requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define requirements and automatically generate and run agent configurations.
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

Refer to the [`examples`](./examples) directory and [`docs/examples.md`](./docs/examples.md) for detailed examples and advanced use-cases.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Focus on simplicity and ease of use, minimizing overhead.
*   **Modular & Configurable:** Allows for easy customization and integration of new components.
*   **Open-Source & Cost-Effective:** Promotes accessibility and affordability for various applications.

### Core Features

*   **Built on openai-agents:** Leverages the foundation of the [openai-agents](https://github.com/openai/openai-agents-python) SDK, enabling streaming, tracing, and agent-loop capabilities. Ensures compatibility with `responses` and `chat.completions` APIs for broad model support.
*   **Fully Asynchronous:** Offers high-performance and efficient execution, especially beneficial for benchmark evaluations.
*   **Tracing & Analysis System:** Offers an in-depth analysis of tool calls and agent trajectories through the `DBTracingProcessor` system (coming soon).

### Automation

*   **YAML-Based Configuration:** Provides structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Generates agent configurations automatically based on user requirements.
*   **Tool Generation & Optimization:** Supports tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep / Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:** Creates web pages based on user inputs.
*   **Trajectory Collection:** Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides value to various user groups:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** that outperforms basic ReAct, serving as a starting point for model training and ablation studies.
*   **One-click evaluation scripts** streamline experimentation and ensure consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:** Quickly get started with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:** Offers encapsulated and highly customizable key components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** Development and debugging are intuitive and straightforward with a rich toolset and visual tracing tools.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can utilize.
*   **Environment:** The operational context of the agent (e.g., browser, shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For further details, consult the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides complete code and examples for quick setup. Follow these steps to run your first agent or use [`docker/README.md`](./docker/README.md) for a Docker-based setup with a frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. We recommend [uv](https://github.com/astral-sh/uv) for dependency management.

Ensure Python and uv are installed.

Then, clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys after copying
```

Fill the required API keys into the `.env` file, like your LLM API keys:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace the API key in the .env file:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker setup.

### Quick Start

Youtu-agent has built-in configurations. For example, the default config (`configs/agents/default.yaml`) defines a simple agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run an interactive CLI chatbot with this agent:

```bash
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search
# (Will be replaced with free alternatives)
python scripts/cli_chat.py --stream --config default
# Run without search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Several ready-to-use examples are available. Configure tool APIs in the `.env` file:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image about “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

To visualize the agent’s runtime status, download and install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.6/utu_agent_ui-0.1.6-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.6-py3-none-any.whl
```

Then run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project using the local link after the terminal shows the successful deployment message:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent automatically searches the web for information and outputs an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking. To evaluate on `WebWalkerQA`:

```bash
# Prepare dataset and save to DB
python scripts/data/process_web_walker_qa.py

# Run evaluation, replace <your_exp_id>
# Set JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, and JUDGE_LLM_API_KEY in .env
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Review results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Explore the framework through the full documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get started quickly.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project utilizes code from the following open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions! Review our [**Contributing Guidelines**](./CONTRIBUTING.md).

## 📚 Citation

Cite this work:

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