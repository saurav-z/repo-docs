# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

**Youtu-Agent** is a flexible, high-performance agent framework that lets you create, run, and evaluate autonomous agents, leveraging the power of open-source models. [Explore the Youtu-Agent Repo](https://github.com/Tencent/Youtu-agent)

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

Youtu-Agent empowers you to build sophisticated agents for data analysis, file processing, and research, all without relying on expensive, closed-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **High-Performance Open-Source:** Achieve strong performance with open-source models like DeepSeek-V3, optimized for cost-effective deployment.
*   ✅ **Practical Use Cases:** Includes out-of-the-box support for CSV analysis, literature review, file organization, and more.
*   ✅ **Flexible & Extensible:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supports various model APIs and tool integrations.
*   ✅ **Automated Workflow:** Simplify agent creation and management with YAML-based configurations and automatic agent generation.
*   ✅ **Verified Performance:** Achieved impressive results, including 71.47% on WebWalkerQA (pass@1) and 72.8% on GAIA (text-only, pass@1).

## 📢 News

*   🎁 \[2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 \[2025-08-28] We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and tools to deliver excellent results on challenging benchmarks.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 71.47% accuracy with `DeepSeek-V3.1`.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images to watch video demonstrations:

| Example                 | Description                                    |
| ----------------------- | ---------------------------------------------- |
| **Data Analysis**       | Analyzes CSV and generates an HTML report.       |
| ![Data Analysis Video](https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775)  | **File Management**      | Renames and categorizes local files.              |
| ----------------------- | ---------------------------------------------- |
| ![File Management Video](https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e) | **Wide Research**   | Gathers comprehensive research for a report. |
| ----------------------- | ---------------------------------------------- |
| ![Wide Research Video](https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92) | **Paper Analysis** | Parses, analyzes, and compiles related literature. |
| ----------------------- | ---------------------------------------------- |
| ![Paper Analysis Video](https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d)    |                  |                                            |

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic generation using YAML configurations.  A "meta-agent" interacts with you to define requirements, then generates and saves the config automatically.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

| Automatic Agent Generation                                                        |
| --------------------------------------------------------------------------------- |
| Interactively define requirements, automatically generate and run the agent config. |
| ![Automatic Agent Generation Video](https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef)                                               |

Explore more examples in the [`examples`](./examples) directory and detailed documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Focus on simplicity and ease of use.
*   **Modular & Configurable:** Customize and integrate new components easily.
*   **Open-Source & Low-Cost:** Promote accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories with `DBTracingProcessor` (will be released soon).

### Automation

*   **YAML-Based Configuration:** Organize agent settings effectively.
*   **Automatic Agent Generation:** Auto-generate agent configurations based on user input.
*   **Tool Generation & Optimization:** Future support for tool evaluation and optimization.

### Use Cases

*   **Deep/Wide Research:** Comprehensive search-oriented tasks.
*   **Webpage Generation:** Generate web pages based on specific inputs.
*   **Trajectory Collection:** Support data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

**Benefit for different User Groups:**

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** that is stronger than basic ReAct, serving as an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases**: Includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability**: A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools.
*   **Environment:** The world in which the agent operates.
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset.

Refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for more details.

## 🚀 Getting Started

Youtu-Agent provides complete code and examples to help you get started quickly. Follow the steps below to run your first agent, or refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with interactive frontend.

### Setup

#### Source Code Deployment

> \[!NOTE]
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

Youtu-agent ships with built-in configurations. For example, the default config (`configs/agents/default.yaml`) defines a simple agent equipped with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

You can launch an interactive CLI chatbot with this agent by running:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

The repository provides multiple ready-to-use examples. Some examples require the agent to have internet search capabilities, so you’ll need to configure the tool APIs in the `.env` file under the tools module:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

For example, to enable the agent to automatically search online for information and generate an SVG image on the topic of “DeepSeek V3.1 New Features,” run the following command:

```bash
python examples/svg_generator/main.py
```

If you want to visualize the agent’s runtime status using the web UI, download the frontend package from the Youtu-Agent releases and install it locally:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.6/utu_agent_ui-0.1.6-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.6-py3-none-any.whl
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

This project is built on the contributions of the following open-source projects:
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