# Youtu-Agent: Build Powerful Agents with Open-Source Models

Youtu-Agent is a cutting-edge agent framework that empowers you to build and deploy autonomous agents, delivering impressive performance with open-source models.  Explore the [Youtu-Agent GitHub Repository](https://github.com/TencentCloudADP/youtu-agent) for more details.

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

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **High Performance:** Achieved state-of-the-art results on WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1, text-only subset) using only DeepSeek-V3 series models, demonstrating strong open-source capabilities.
*   💡 **Open-Source & Cost-Effective:** Designed for accessible, low-cost deployment, eliminating reliance on expensive, closed-source models.
*   🛠️ **Practical Use Cases:** Out-of-the-box support for diverse tasks like data analysis, file processing, and deep research, with more capabilities coming soon.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), and supports a wide variety of model APIs and tool integrations.
*   🤖 **Simplified Automation:** YAML-based configurations and automatic agent generation streamline setup and reduce manual effort.

## 🗞️ News

*   🎁 **[Sep 2, 2025]:** Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) with `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[Aug 28, 2025]:** Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [Documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent is built on open-source models, achieving competitive results on challenging benchmarks.

*   **WebWalkerQA:** Achieved 71.47% accuracy with DeepSeek-V3.1, setting a new SOTA.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore practical applications of Youtu-Agent with these interactive examples:

|                                                                                                      |                                                                                                      |
| :--------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Data Analysis:** Analyzes a CSV and generates an HTML report.                                       | **File Management:** Renames and categorizes local files.                                             |
| <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920"  poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg"  controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e"  poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg"  controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| **Wide Research:**  Gathers information to generate a comprehensive report.                             | **Paper Analysis:**  Parses, analyzes, and compiles related literature.                                |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92"  poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg"  controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d"  poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg"  controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation:

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

|                                                                                                    |
| :------------------------------------------------------------------------------------------------- |
| **Automatic Agent Generation:**  Interactively define requirements, automatically generate the config, and run. |
| <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef"  poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg"  controls muted preload="metadata" width="100%" height="auto" style="object-fit: cover; border-radius: 8px;"></video> |

For more examples and detailed documentation, see the [`examples`](./examples) directory and [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
*   **Minimal Design:** Keep the framework simple.
*   **Modular & Configurable:** Flexible customization.
*   **Open-Source & Low-Cost:** Promotes accessibility.

### Core Features
*   **Built on openai-agents:** Leverage openai-agents with streaming, tracing, and agent-loop capabilities.
*   **Fully Asynchronous:** High-performance execution, especially for benchmarks.
*   **Tracing & Analysis System:**  In-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation
*   **YAML Based Configuration:** Easily manage agent configurations.
*   **Automatic Agent Generation:** Automatically create agent configurations based on requirements.
*   **Tool Generation & Optimization:** Tool evaluation, automated optimization, and customized tool generation will be supported in the future.

### Use Cases
*   **Deep / Wide research**: Standard search-oriented tasks.
*   **Webpage generation**: Generate webpages based on inputs.
*   **Trajectory collection**: Data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides value for various users:

### For Researchers & LLM Trainers
*   A **simple yet powerful baseline** for model training and ablation studies.
*   **One-click evaluation scripts** to streamline the experimental process.

### For Agent Application Developers
*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts.
*   **Modular Design**: Key components are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts
*   **Practical Use Cases**: Includes tasks like deep research, data analysis, and file organization.
*   **Simplicity & Debuggability**: A rich toolset makes development and debugging intuitive.

## 🧩 Core Concepts

*   **Agent**: An LLM with prompts, tools, and an environment.
*   **Toolkit**: An encapsulated set of tools.
*   **Environment**: Where the agent operates (e.g., browser, shell).
*   **ContextManager**: Module for managing the agent's context window.
*   **Benchmark**: Workflow for a specific dataset, including logic.

More details can be found in our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Get started quickly with the following steps. Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # NOTE: You should then config the necessary API keys.
    ```

3.  Populate the `.env` file with your API keys (e.g., LLM, search).

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

Refer to [`docker/README.md`](./docker/README.md) for Docker setup.

### Quick Start

Run a simple agent with a search tool:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

To enable internet search capabilities, configure tool APIs in `.env`.  For example:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Run an example:

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

Access the project via the local link once the terminal shows the following message:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on WebWalkerQA:

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

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Detailed guide.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Common questions and issues.

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Read our [**Contributing Guidelines**](./CONTRIBUTING.md) if you want to help improve Youtu-Agent.

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