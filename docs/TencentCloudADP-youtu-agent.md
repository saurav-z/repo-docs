# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Unleash the power of autonomous agents with Youtu-Agent, a high-performance framework that delivers cutting-edge capabilities with accessible, open-source models.**  Explore the [Youtu-Agent GitHub repository](https://github.com/TencentCloudADP/youtu-agent).

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

Youtu-Agent is a versatile framework designed for building, running, and evaluating autonomous agents. It excels in tasks like data analysis, file processing, and research, utilizing open-source models for cost-effective deployment and impressive performance.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **High-Performance Benchmarks:** Achieved impressive results (71.47% on WebWalkerQA, 72.8% on GAIA text-only subset, pass@1) with DeepSeek-V3 series models.
*   ✅ **Cost-Effective & Open-Source Focused:** Optimized for accessible deployment without relying on closed models.
*   ✅ **Practical Use Cases:** Supports CSV analysis, literature review, file organization, and more.
*   ✅ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), with extensibility for various models (DeepSeek, gpt-oss) and tool integrations.
*   ✅ **Simplified Automation:** YAML-based configuration, automatic agent generation, and streamlined setup reduce development time.

## 🗞️ News

*   🎁 **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]** We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent delivers strong performance on challenging deep search and tool use benchmarks, leveraging open-source models and lightweight tools.

*   **WebWalkerQA:** Achieved 60.71% accuracy with `DeepSeek-V3-0324`, improving to **71.47%** with the new `DeepSeek-V3.1`.
*   **GAIA:** Achieved **72.8%** pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) with `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the capabilities of Youtu-Agent with these interactive examples. Click the images to view detailed videos.

|                                                                                                                                  |                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Data Analysis**<br>Analyzes a CSV file and generates an HTML report.                                                        | **File Management**<br>Renames and categorizes local files.                                                                            |
| <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920"  poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg"  controls muted preload="metadata"  width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e"  poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg"  controls muted preload="metadata"  width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| **Wide Research**<br>Gathers comprehensive information to generate a detailed report.                                         | **Paper Analysis**<br>Parses a paper, performs analysis, and compiles related literature.                                              |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92"  poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg"  controls muted preload="metadata"  width="100%" height=300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d"  poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg"  controls muted preload="metadata"  width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |

> [!NOTE]
> Explore more examples in the [`examples`](./examples) directory and the detailed [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its automatic agent generation feature.  Define agent requirements interactively, and the framework will automatically generate and save the configuration using simple YAML-based configs.

```bash
# Interactively define your requirements and automatically generate the config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

|                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------- |
| **Automatic Agent Generation**<br>Interactively clarify your requirements, generate the agent configuration, and run it immediately. |
| <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef"  poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg"  controls muted preload="metadata"  width="100%" height="auto"  style="object-fit: cover; border-radius: 8px;"></video> |

> [!NOTE]
> Learn more about automatic agent generation in the [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Focus on simplicity and ease of use.
*   **Modular and Configurable:**  Easily customize and integrate new components.
*   **Open-Source Model Support & Low Cost:** Promote accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leverages the power of the [openai-agents](https://github.com/openai/openai-agents-python) SDK, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories using our `DBTracingProcessor` system. (will be released soon)

### Automation

*   **YAML-based configuration:** Manage agents with structured configuration files.
*   **Automatic agent generation:**  Generate agent configurations based on user requirements.
*   **Tool generation & optimization:** Future support for tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep/Wide Research:** Search-oriented tasks.
*   **Webpage Generation:** Examples include generating web pages based on specific inputs.
*   **Trajectory Collection:** Collect data for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant advantages for various users:

### For Agent Researchers & LLM Trainers

*   **Strong Baseline:** Provides a solid foundation, outperforming ReAct, for model training and ablation studies.
*   **One-Click Evaluation:** Streamlines the experimental process for consistent benchmarking.

### For Agent Application Developers

*   **Proven Scaffolding:** A reliable starting point for building agent applications.
*   **Ease of Use:** Get started quickly with user-friendly scripts and a rich toolkit.
*   **Modular Design:**  Customize core components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Examples:** Explore tasks like deep research, data analysis, and file organization in the `/examples` directory.
*   **Simplicity & Debuggability:** Intuitive development with a comprehensive toolset and visual tracing.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** A collection of tools an agent can use.
*   **Environment:** Where the agent operates (e.g., a browser).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** A structured workflow for a dataset, including preprocessing and evaluation.

Refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for more details on design and implementation.

## 🚀 Getting Started

Youtu-Agent offers complete code and examples for a quick start.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+.  Use [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Ensure Python and uv are installed.

2.  Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure API keys here.
    ```

3.  Populate the `.env` file with necessary API keys (LLM, etc.).

    ```bash
    # Example LLM configuration:
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    For DeepSeek API access, take advantage of the free tokens offer:

    ```bash
    # DeepSeek API configuration (Tencent Cloud Free Tokens)
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup with an interactive frontend.

### Quick Start

Use the default configuration to launch an interactive CLI chatbot:

```bash
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --stream --config default
# To avoid the search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs (SERPER_API_KEY, JINA_API_KEY) in the `.env` file under the tools module for web search.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image on “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

To use the web UI, download and install the frontend package:

```bash
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link (e.g., `http://127.0.0.1:8848/`).

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will automatically search the web and output an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

To evaluate on `WebWalkerQA`:

```bash
# Prepare the dataset.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` and a custom `exp_id`.
# Ensure JUDGE_LLM_* keys are set in .env.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Explore the framework in detail:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Detailed setup and usage guide.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the work of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions!  Read our [**Contributing Guidelines**](./CONTRIBUTING.md).

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