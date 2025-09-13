# Youtu-Agent: Build and Deploy Powerful Agents with Open-Source Models

[<img src="docs/assets/logo.svg" alt="Youtu-agent Logo" height="24px">](https://github.com/TencentCloudADP/youtu-agent) Youtu-Agent is a flexible and efficient framework that empowers you to create and run autonomous agents, even with open-source models, achieving remarkable performance and cost-effectiveness.

<div align="center">
  <a href="https://tencentcloudadp.github.io/youtu-agent/"><img src="https://img.shields.io/badge/📖-Documentation-blue.svg" alt="Documentation"></a>
  <a href="https://github.com/TencentCloudADP/youtu-agent"><img src="https://img.shields.io/badge/GitHub-Tencent-blue.svg" alt="GitHub"></a>
  <a href="https://deepwiki.com/TencentCloudADP/youtu-agent"><img src="https://img.shields.io/badge/DeepWiki-Tencent-blue.svg" alt="DeepWiki"></a>
</div>

<p align="center">
  <a href="README_ZH.md"><b>中文</b></a> |
  <a href="README_JA.md"><b>日本語</b></a> |
  <a href="#-benchmark-performance"><b>🌟 Performance</b></a> |
  <a href="#-examples"><b>💡 Examples</b> </a> |
  <a href="#-features"><b>✨ Features</b> </a> |
  <a href="#-getting-started"><b>🚀 Getting Started</b> </a> |
  <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

**Key Features:**

*   **High Performance with Open-Source:** Achieves state-of-the-art results on benchmarks like WebWalkerQA and GAIA using open-source models, minimizing reliance on costly closed-source solutions.
*   **Cost-Effective Deployment:** Optimized for accessible and budget-friendly agent development without sacrificing performance.
*   **Practical Use Cases:** Supports tasks like data analysis, file processing, literature reviews, and more, out-of-the-box.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), enabling integration with diverse model APIs (DeepSeek, gpt-oss, etc.), tool integrations, and framework implementations.
*   **Automated Agent Generation:** Simplifies agent creation with YAML-based configurations and automated generation, minimizing manual setup.

## News

*   🎁 **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]** We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent demonstrates impressive performance on challenging benchmarks using open-source models and lightweight tools.

*   **WebWalkerQA:** Achieved 71.47% pass@1 with DeepSeek-V3.1, setting a new state-of-the-art (SOTA).
*   **GAIA:** Reached 72.8% pass@1 on the text-only subset using DeepSeek-V3-0324.  Full GAIA benchmark evaluation with multimodal tools is planned.

![WebWalkerQA Benchmark](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the power of Youtu-Agent through a variety of practical examples. Click the images to watch detailed videos.

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyze a CSV file and generate an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Rename and categorize local files.
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
      <strong>Wide Research</strong><br>Generate a comprehensive report based on extensive information.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parse a paper, analyze it, and compile related literature.
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

### 🤖 Automatic Agent Generation

A standout feature of `Youtu-Agent` is its ability to **automatically generate agent configurations**.  This streamlines the process of agent creation with intuitive YAML-based configurations.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define requirements and automatically generate the agent configuration.
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

For more in-depth examples, explore the [`examples`](./examples) directory and the detailed documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![Features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeping the framework simple and easy to use.
*   **Modular & Configurable:** Flexible customization and easy integration.
*   **Open-Source & Cost-Effective:** Promotes accessibility and affordability.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories using our `DBTracingProcessor` system (coming soon).

### Automation

*   **YAML-Based Configuration:** Manage agent configurations with structured YAML files.
*   **Automatic Agent Generation:** Automatically generate agent configurations based on user input.
*   **Tool Generation & Optimization:** Future support for automated tool evaluation, optimization, and customized tool generation.

### Use Cases

*   Deep / Wide Research
*   Webpage Generation
*   Trajectory Collection

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides value for a variety of users:

### For Agents Researchers & LLM Trainers

*   A strong baseline for model training and research.
*   One-click evaluation scripts for streamlining experiments.

### For Agent Application Developers

*   A proven and portable framework for building agent applications.
*   Ease of Use with simple scripts and built-in toolkits.
*   Modular Design with customizable components.

### For AI & Agent Enthusiasts

*   Practical Use Cases available in the `/examples` directory.
*   Simplicity and Debuggability with visual tracing tools.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more details on design and implementation, see the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Get started quickly with Youtu-Agent using these steps, or refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker setup.

### Setup

#### Source Code Deployment

> [!NOTE]
> This project requires Python 3.12+.  We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

First, ensure Python and `uv` are installed.

Then, clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  #  Configure API keys here.
```

Populate your `.env` file with the required API keys, such as LLM keys:

```bash
# LLM configuration (OpenAI format compatibility)
# Setup your LLM config, ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free.  After you apply, replace the API key in `.env`:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with an interactive frontend.

### Quick Start

Youtu-Agent includes built-in configurations. The default config (`configs/agents/default.yaml`) defines a simple agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Launch an interactive CLI chatbot with this agent using:

```bash
# Configure SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (Free alternatives are planned for the future)
python scripts/cli_chat.py --stream --config default
# To run without the search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Numerous ready-to-use examples are provided. Configure tool APIs in `.env` for examples requiring internet search:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Get API Key from the URL in the comments>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Get API Key from the URL in the comments>
```

For example, to have the agent search for "DeepSeek V3.1 New Features" and generate an SVG image:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime status using a web UI, download the frontend package from Youtu-Agent releases and install it:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then, run the web version of the SVG image generation:

```bash
python examples/svg_generator/main_web.py
```

Open the local link after the terminal displays "Server started at http://127.0.0.1:8848/".

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will automatically search the web, collect information, and output an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking on standard datasets. For `WebWalkerQA`:

```bash
# Prepare dataset (downloads and processes WebWalkerQA, saves to DB)
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml`, your custom `exp_id`, and the sampled dataset `WebWalkerQA_15`.
# NOTE: Set `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored for analysis in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Explore Youtu-Agent further with these resources:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Discover core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide for getting started.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the contributions of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Contributions are welcome! Read our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

## 📚 Citation

If you use this work, please cite:

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