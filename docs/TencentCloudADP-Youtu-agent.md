<!--
  _   _   _       _   _   _   _   _   _   _
 / \ / \ / \     / \ / \ / \ / \ / \ / \ / \
( Y | o | u | t | u | - | A | g | e | n | t )
 \_/ \_/ \_/     \_/ \_/ \_/ \_/ \_/ \_/ \_/
-->

# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

> **Youtu-Agent is a flexible and high-performance framework that empowers you to create autonomous agents, offering data analysis, file processing, and research capabilities, all powered by open-source models. Explore the power of Youtu-Agent on [GitHub](https://github.com/TencentCloudADP/Youtu-agent)!**

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

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="right" style="margin-left:20px;">

Youtu-Agent is a versatile and efficient framework designed for constructing, executing, and assessing autonomous agents. It excels in diverse tasks such as data analysis, file handling, and in-depth research, all while leveraging open-source models, making AI agent development accessible and cost-effective.

## Key Features:

*   **Superior Performance:** Achieved state-of-the-art results on WebWalkerQA (71.47% pass@1) and GAIA (72.8% on text-only subset, pass@1) using DeepSeek-V3 series models, demonstrating a strong foundation for open-source agent development.
*   **Cost-Effective & Open Source:** Designed for affordable deployment without reliance on proprietary models, fostering a more accessible approach to AI agent creation.
*   **Versatile Use Cases:** Out-of-the-box support for practical tasks including CSV analysis, literature reviews, personal file organization, and upcoming support for podcast/video generation.
*   **Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), ensuring compatibility with diverse model APIs (DeepSeek, gpt-oss, etc.) and tool integrations.
*   **Simplified Automation:** YAML-based configuration, automatic agent generation, and streamlined setup significantly reduce manual effort, accelerating the development cycle.

## 🗞️ News

*   🎁 **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]** We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and lightweight tools to achieve impressive results on demanding deep search and tool use benchmarks.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 60.71% accuracy with `DeepSeek-V3-0324`, and the latest `DeepSeek-V3.1` further improved to 71.47%, demonstrating cutting-edge performance.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Scored 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324` (including models used within tools). We are actively extending evaluation to the full GAIA benchmark with multimodal tools, and will release the trajectories in the near future. Stay tuned! ✨

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
      <video src="https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775"
             poster="https://img.youtube.com/vi/SCR4Ru8_h5Q/sddefault.jpg"
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

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its automated configuration generation feature. Instead of coding or complex prompt engineering, define your agent using straightforward YAML configuration. A built-in "meta-agent" streamlines the process by interacting with you to capture requirements, then automatically generates and saves the configuration.

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

Explore more examples and advanced use cases in the [`examples`](./examples) directory and comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeps the framework straightforward and user-friendly, avoiding unnecessary complexity.
*   **Modular & Configurable:** Provides flexibility for customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost:** Focuses on accessibility and cost-effectiveness for various applications.

### Core Features

*   **Built on openai-agents:** Leveraging the robust [openai-agents](https://github.com/openai/openai-agents-python) SDK, the framework inherits essential capabilities like streaming, tracing, and agent-loop, ensuring compatibility with `responses` and `chat.completions` APIs for seamless model adaptation (e.g., gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution, particularly beneficial for benchmark evaluations.
*   **Tracing & Analysis System:** The `DBTracingProcessor` offers detailed analysis of tool calls and agent trajectories, enhancing debugging and understanding (coming soon).

### Automation

*   **YAML-Based Configuration:** Structured and manageable agent configurations.
*   **Automatic Agent Generation:** Facilitates automated configuration creation based on user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, automated optimization, and custom tool generation.

### Use Cases

*   **Deep / Wide Research:** Supports common search-oriented tasks.
*   **Webpage Generation:** Includes examples of generating web pages based on specific inputs.
*   **Trajectory Collection:** Provides capabilities for data collection to aid in training and research efforts.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant advantages for diverse user groups:

### For Agents Researchers & LLM Trainers

*   A **straightforward baseline** surpasses basic ReAct, providing an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** streamline the experimental process and ensure reliable benchmarking.

### For Agent Application Developers

*   A **proven and adaptable scaffolding** to facilitate building real-world agent applications.
*   **Ease of Use:** Get up and running quickly with simple scripts and a rich suite of built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** The `/examples` directory features applications like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A comprehensive toolkit and visual tracing tools enhance development and debugging with intuitive methods.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated collection of tools available to the agent.
*   **Environment:** The operational context of the agent (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For detailed design and implementation insights, consult the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides comprehensive code and examples to facilitate a rapid start. Follow the steps below to run your first agent or refer to [`docker/README.md`](./docker/README.md) for a simplified Docker setup with an interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

First, make sure Python and uv are installed.

Then clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

After copying the `.env.example` file, you need to fill in the necessary keys in the `.env` file, e.g. LLM API keys. For example:

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

Youtu-agent includes pre-built configurations. The default configuration (`configs/agents/default.yaml`) defines a simple agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run an interactive CLI chatbot with this agent using:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

The repository features multiple ready-to-use examples. Certain examples necessitate internet search capabilities; therefore, configuring tool APIs in the `.env` file under the tools module is required:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To enable the agent to automatically search the web and produce an SVG image centered on “DeepSeek V3.1 New Features,” run the command:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime status using the web UI, download the frontend package from the Youtu-Agent releases and install it locally:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```

Next, execute the web version of the SVG image generation:

```bash
python examples/svg_generator/main_web.py
```

The deployment is confirmed when the terminal displays the following message, allowing project access via the local link:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will automatically search the web, gather relevant information, and generate an SVG visualization upon a research topic.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent also supports benchmarking on standard datasets. For example, to evaluate on `WebWalkerQA`:

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

## 🙏 Acknowledgements

This project is built upon the contributions of several key open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

If you find this work useful, please consider citing:

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