<!-- README.md -->
# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Unlock the power of autonomous agents with Youtu-Agent, a flexible and high-performance framework that delivers state-of-the-art results using open-source models.**  [Explore the original repository on GitHub](https://github.com/TencentCloudADP/Youtu-agent)

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

[中文](README_ZH.md) | [日本語](README_JA.md) | [🌟 Performance](#-benchmark-performance) | [💡 Examples](#-examples) | [✨ Features](#-features) | [🚀 Getting Started](#-getting-started) | [📢 Join Community](https://discord.gg/svwuqgUx)

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="right" style="margin-left:20px;">

Youtu-Agent provides a robust and versatile framework for creating, running, and evaluating autonomous agents.  Built with a focus on open-source models and cost-effectiveness, it offers exceptional performance across diverse tasks.

**Key Features:**

*   ✅ **High Performance:** Achieved impressive scores on WebWalkerQA (71.47%) and GAIA (72.8%), demonstrating strong capabilities with open-source models like DeepSeek-V3.
*   💰 **Cost-Effective & Open-Source:** Designed for efficient deployment with accessible, open-source models, minimizing costs.
*   ⚙️ **Flexible Architecture:**  Built on [openai-agents](https://github.com/openai/openai-agents-python), supports diverse model APIs (DeepSeek, gpt-oss) and tool integrations.
*   🚀 **Automated Configuration:** YAML-based configurations and automatic agent generation streamline development and reduce manual effort.
*   🛠️ **Practical Use Cases:**  Ready-to-use examples for data analysis, file management, research, and more, with ongoing expansion.

## 🗞️ News

*   🎁 **[September 2025]** Tencent Cloud International offers new DeepSeek API users **3 million free tokens** (September 1 – October 31, 2025).  [Get free tokens](https://www.tencentcloud.com/document/product/1255/70381) to use DeepSeek models in Youtu-Agent!  Also, check out the [Agent Development Platform](https://adp.tencentcloud.com) (ADP) for enterprise agent solutions.
*   📺 **[August 2025]**  Live sharing updates on DeepSeek-V3.1 and its integration with the Youtu-Agent framework. [See documentation here](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels on benchmarks using open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA):** Achieved **71.47%** accuracy with DeepSeek-V3.1, establishing a new state-of-the-art.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/):**  Scored **72.8%** (pass@1) on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using DeepSeek-V3-0324 (including tools).  Multimodal tool evaluation is in progress.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

See Youtu-Agent in action. Click the images below for detailed videos showcasing real-world agent applications.

| Data Analysis                                                    | File Management                                                   |
| :--------------------------------------------------------------- | :---------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920" poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| **Analyzes a CSV file and generates an HTML report.**              | **Renames and categorizes local files for the user.**               |

| Wide Research                                                    | Paper Analysis                                                     |
| :--------------------------------------------------------------- | :---------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| **Gathers extensive information to generate a comprehensive report.**   | **Parses, analyzes, and compiles related literature.**             |

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with **automatic agent generation** using YAML-based configurations.  A "meta-agent" interacts with you to capture requirements and generates the configuration automatically.

```bash
# Interactively define requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

| Automatic Agent Generation                                                                               |
| :------------------------------------------------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" controls muted preload="metadata" width="100%" height="auto" style="object-fit: cover; border-radius: 8px;"></video> |
| **Clarify requirements interactively, automatically generate agent configuration, and run immediately.** |

For more detailed examples and advanced use cases, explore the [`examples`](./examples) directory and the documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Focus on simplicity and ease of use.
*   **Modular & Configurable:**  Customizable with easy integration of new components.
*   **Open-Source & Low-Cost:** Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, ensuring compatibility with `responses` and `chat.completions` APIs, and diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories (coming soon).

### Automation

*   **YAML-Based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Agent configurations are automatically generated based on user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep/Wide Research:** Supports common search-oriented tasks.
*   **Webpage Generation:** Generate web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides significant value for various user groups:

### For Agents Researchers & LLM Trainers

*   **Simple, Powerful Baseline:**  Stronger than basic ReAct, ideal as a starting point for model training and ablation studies.
*   **One-Click Evaluation Scripts:** Streamlines the experimental process for consistent benchmarking.

### For Agent Application Developers

*   **Proven & Portable Scaffolding:** For building real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated and customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Explore tasks in the `/examples` directory such as deep research, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** Intuitive development with a rich toolset and visual tracing tools.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can utilize.
*   **Environment:** The world the agent operates within (e.g., browser, shell).
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more details, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent offers comprehensive code and examples to help you quickly start your agent development journey. Follow the instructions below, or use the Docker setup at [`docker/README.md`](./docker/README.md) for a simplified setup with an interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> This project requires Python 3.12+.  We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

Ensure Python and `uv` are installed. Then, clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys in .env file.
```

Fill in the `.env` file with your required API keys (LLM API keys). For example:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) is offering new DeepSeek API users **3 million free tokens** (September 1 – October 31, 2025). [Apply here](https://www.tencentcloud.com/document/product/1255/70381). After you've applied, update the `.env` file:

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

Youtu-Agent comes with built-in configurations, such as `configs/agents/default.yaml`:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Launch a CLI chatbot with:

```bash
# Configure SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (Alternatives planned for the future)
python scripts/cli_chat.py --stream --config default
# To avoid the search toolkit, run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs (e.g., search) in the `.env` file:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To generate an SVG image, run:

```bash
python examples/svg_generator/main.py
```

To visualize agent runtime status with the web UI, download the frontend package from Youtu-Agent releases:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then, run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link after successful deployment: `http://127.0.0.1:8848/`

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will search the web, collect information, and output an SVG visualization:

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking on standard datasets (e.g., WebWalkerQA):

```bash
# Prepare dataset. Downloads and processes WebWalkerQA dataset.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml`. Specify `exp_id` and dataset.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Expand your knowledge with our comprehensive documentation:

-   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
-   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get started quickly.
-   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project leverages:
-   [openai-agents](https://github.com/openai/openai-agents-python)
-   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
-   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

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