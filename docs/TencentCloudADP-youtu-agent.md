# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework that empowers you to create, run, and evaluate autonomous agents using open-source models, offering a cost-effective and accessible AI agent solution. Explore the [original repository](https://github.com/TencentCloudADP/youtu-agent) for more details.

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#examples"><b>💡 Examples</b> </a>
| <a href="#features"><b>✨ Features</b> </a>
| <a href="#getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **High Performance:** Achieved impressive results on WebWalkerQA (71.47%) and GAIA (72.8%), demonstrating strong capabilities with open-source `DeepSeek-V3` models.
*   💰 **Cost-Effective & Open-Source Friendly:** Designed for accessible and affordable deployment without relying on expensive proprietary models.
*   🛠️ **Practical Use Cases:** Supports tasks like data analysis, file processing, literature review, and more, with new applications (like podcast and video generation) coming soon.
*   ⚙️ **Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), allowing easy integration with diverse models, tool APIs, and framework implementations.
*   ✨ **Automation & Simplicity:** Streamline agent creation with YAML-based configurations and automatic agent generation.

## 📰 News

*   📺 [2025-09-09] Live sharing of design philosophy and usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 [2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 [2025-08-28] Live sharing of updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent excels on challenging benchmarks with open-source models and lightweight tools, showcasing its potential for both research and practical applications.

*   **WebWalkerQA:** Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art.
*   **GAIA (text-only subset):** Scored 72.8% pass@1 using `DeepSeek-V3-0324`. Multimodal tool support is planned for the future.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore these example use cases to see Youtu-Agent in action:

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

> [!NOTE]
> Explore more examples in the [`examples`](./examples) directory and comprehensive [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automatic Agent Generation

Youtu-Agent revolutionizes agent creation with automatic configuration generation: simply describe your agent's requirements, and it generates the configuration automatically!

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Describe your needs and Youtu-Agent will build the agent configuration.
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
> See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Simple and easy to use, without unnecessary overhead.
*   **Modular & Configurable:** Enables customization and easy integration of new components.
*   **Open-Source & Low-Cost:** Makes AI agent development accessible and affordable.

### Core Features

*   **Built on OpenAI-Agents:** Leveraging the power of [openai-agents](https://github.com/openai/openai-agents-python) for streaming, tracing, and agent-loop functionalities. Compatible with `responses` and `chat.completions` APIs.
*   **Fully Asynchronous:** Ensures high performance and efficient execution, critical for benchmarking.
*   **Tracing & Analysis System:** Provides detailed analysis of tool calls and agent trajectories with our `DBTracingProcessor` system (coming soon).

### Automation

*   **YAML-Based Configuration:** Simplifies agent management with structured configurations.
*   **Automatic Agent Generation:** Automates configuration based on user requirements.
*   **Tool Generation & Optimization:** Automated tool evaluation, optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep/Wide Research:** Supports common research-oriented tasks.
*   **Webpage Generation:** Includes examples of webpage generation based on input.
*   **Trajectory Collection:** Enables data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides valuable benefits for different user groups:

### For Agents Researchers & LLM Trainers

*   **Strong Baseline:** A simple yet powerful baseline exceeding basic ReAct, perfect for model training.
*   **One-Click Evaluation:** Streamline experiments with easy-to-use evaluation scripts.

### For Agent Application Developers

*   **Proven Scaffolding:** Build real-world agent applications with reliable, portable code.
*   **Ease of Use:** Quick start with simple scripts and a rich toolset.
*   **Modular Design:** Highly customizable key components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Explore diverse tasks like research report generation and data analysis.
*   **Simplicity & Debuggability:** Simplify development and debugging with a comprehensive toolset.

## 🧩 Core Concepts

*   **Agent:** An LLM with defined prompts, tools, and an environment.
*   **Toolkit:** A set of encapsulated tools for the agent.
*   **Environment:** The context where the agent operates (e.g., a browser).
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** A workflow for a specific dataset, with preprocessing and judging logic.

For more technical details, see our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to quickly run your first agent. Or, use [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and we recommend [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and `uv`.
2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure necessary API keys
```

Fill the `.env` file with your API keys (e.g., LLM API keys):

```bash
# LLM config, ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace the key in `.env`:

```bash
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

See [`docker/README.md`](./docker/README.md) for a streamlined Docker setup.

### Quick Start

Use the default agent configuration:

```bash
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --stream --config default

# Without search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 Learn more: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` for examples requiring internet search:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Get the API Key>
```

Run the SVG image generation example:

```bash
python examples/svg_generator/main.py
```

Visualize the agent in the web UI (install the frontend first):

```bash
# Download the frontend
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

# Run the web example
python examples/svg_generator/main_web.py
```

Access the project via the local link shown in the terminal.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on datasets like `WebWalkerQA`:

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Evaluate (set JUDGE_LLM_* in .env)
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get up and running quickly.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project is built upon these open-source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome community contributions! See our [**Contributing Guidelines**](./CONTRIBUTING.md).

## 📚 Citation

If you find this work useful, please cite:

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