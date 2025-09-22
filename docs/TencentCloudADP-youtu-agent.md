# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

Youtu-Agent is a flexible, high-performance framework enabling you to build, run, and evaluate intelligent agents powered by open-source models, unlocking new possibilities in AI. ([See the original repo](https://github.com/TencentCloudADP/youtu-agent))

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
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

**Youtu-Agent** empowers you to create agents for tasks like data analysis, file processing, and research – all while leveraging the power and cost-effectiveness of open-source models.

**Key Features:**

*   **Exceptional Performance:** Achieves SOTA results on benchmarks like WebWalkerQA and GAIA using open-source `DeepSeek-V3` models.
*   **Open-Source & Cost-Effective:** Designed for accessibility and low-cost deployment, avoiding reliance on proprietary models.
*   **Practical Use Cases:** Includes out-of-the-box support for CSV analysis, research, file organization, and more (with expansion planned).
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting diverse model APIs and tool integrations.
*   **Simplified Automation:** Uses YAML-based configurations and automatic agent generation to streamline development.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimalist Design:**  Focus on simplicity and ease of use.
*   **Modular & Configurable:**  Allows for easy customization and integration of new components.
*   **Open-Source Model Support & Cost-Effective:** Promotes accessibility and cost-effectiveness for various applications.

### Core Features

*   **Based on openai-agents**: Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
*   **Tracing & Analysis System**: Provides in-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation

*   **YAML Based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Generate configurations based on user requirements.
*   **Tool Generation & Optimization:** Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide Research**: Covers common search-oriented tasks.
*   **Webpage Generation**: Generate web pages based on specific inputs.
*   **Trajectory Collection**: Supports data collection for training and research purposes.

## 🗞️ News

*   📺 [2025-09-09] Sharing the design philosophy and basic usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 [2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 [2025-08-28] Sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

`Youtu-Agent` showcases strong results on challenging benchmarks using open-source models and lightweight tools:

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 71.47% accuracy with `DeepSeek-V3.1`.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

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

Youtu-Agent's automatic agent generation simplifies the creation of task-specific agents using simple YAML-based configurations.

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

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed to deliver significant value to different user groups:

### For Agents Researchers & LLM Trainers

*   **Strong Baseline:** Serves as an excellent starting point for model training and ablation studies.
*   **One-Click Evaluation:** Streamlines the experimental process and ensures consistent benchmarking.

### For Agent Application Developers

*   **Proven Scaffolding:** Simplifies building real-world agent applications.
*   **Ease of Use:**  Quickly get started with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:**  Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Offers examples like deep research reports and data analysis.
*   **Simplicity & Debuggability:** Development and debugging are made intuitive and straightforward.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can use.
*   **Environment:** The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For in-depth details, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides complete code and examples for quick onboarding.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys here
```

Replace the placeholder API keys in the `.env` file.

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with an interactive frontend.

### Quick Start

Use the `simple/base_search` configuration for a simple agent:

```bash
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in the `.env` file for examples.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Get API Key from Serper>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Get API Key from Jina>
```

To generate an SVG image:

```bash
python examples/svg_generator/main.py
```

Run the web version:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
python examples/svg_generator/main_web.py
```

Access the project at the local link.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

To evaluate on `WebWalkerQA`:

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation with your custom `exp_id`
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

View results in the evaluation platform.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

-   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
-   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get up and running quickly.
-   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project leverages the work of:
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

See our [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute.

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

[![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)](https://star-history.com/#TencentCloudADP/youtu-agent)