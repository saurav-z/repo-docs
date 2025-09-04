# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent is a flexible and high-performing framework enabling you to build, run, and evaluate AI agents using open-source models.**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/Youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

**Key Features:**

*   ✅ **High Performance:** Achieves strong results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%) using open-source DeepSeek models.
*   💡 **Open-Source & Cost-Effective:** Designed for accessible deployment, reducing reliance on expensive closed models.
*   🛠️ **Practical Use Cases:** Includes out-of-the-box support for data analysis, file management, research, and more.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python) and supports diverse models, tools, and frameworks.
*   🚀 **Automation:** YAML-based configurations and automated agent generation streamline setup and management.

## 📰 News

*   **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free!  For enterprise agent solutions, check out [Agent Development Platform](https://adp.tencentcloud.com).
*   **[2025-08-28]** Live sharing updates on DeepSeek-V3.1 and its usage in `Youtu-Agent`, along with shared [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

`Youtu-Agent` demonstrates strong performance on deep search and tool use benchmarks with open-source models and lightweight tools.

*   **WebWalkerQA:** 71.47% accuracy with `DeepSeek-V3.1`.
*   **GAIA (text-only subset):** 72.8% pass@1 with `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click on the images to view detailed videos.

| Data Analysis                                                                   | File Management                                                                    |
| :------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775" poster="https://img.youtube.com/vi/SCR4Ru8_h5Q/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| Wide Research                                                                   | Paper Analysis                                                                     |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" controls muted preload="metadata" width="100%" height=300 style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation. Use YAML-based configs to streamline automation.

```bash
# Interactively define requirements, and auto-generate config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

| Automatic Agent Generation                                                       |
| :-------------------------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" controls muted preload="metadata" width="100%" height="auto" style="object-fit: cover; border-radius: 8px;"></video> |

For advanced use-cases, please refer to the [`examples`](./examples) directory and our comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal design:** User-friendly simplicity.
*   **Modular & configurable:** Easy customization and integration.
*   **Open-source support & low-cost:** Accessible and cost-effective.

### Core Features

*   **Built on openai-agents:** Compatibility with `responses` and `chat.completions` APIs, including [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully asynchronous:** Enables high-performance execution, essential for benchmark evaluations.
*   **Tracing & analysis system:** Provides in-depth analysis of tool calls and agent trajectories via the `DBTracingProcessor`. (will be released soon)

### Automation

*   **YAML-based configuration:** Structured agent configurations.
*   **Automatic agent generation:**  Based on requirements, agents can be automatically configured.
*   **Tool generation & optimization:** Tool evaluation and optimization with customized tool generation supported in the future.

### Use Cases

*   **Deep / Wide research:** Facilitates search-oriented tasks.
*   **Webpage generation:** Creates web pages based on input.
*   **Trajectory collection:** Supports training and research data collection.

## 🤔 Why Choose Youtu-Agent?

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** for model training and ablation studies.
*   **One-click evaluation scripts** to streamline experimental process.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:** Quickstart with simple scripts and built-in toolkits.
*   **Modular Design:** Customizable components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Deep research, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** User-friendly development and debugging tools.

## 🧩 Core Concepts

*   **Agent:**  LLM with specific prompts, tools, and environment.
*   **Toolkit:**  Encapsulated set of tools for the agent.
*   **Environment:**  Where the agent operates (e.g., browser, shell).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:**  Workflow for a specific dataset, including preprocessing and evaluation.

For details, see the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow the steps to get started or use [`docker/README.md`](./docker/README.md) for a streamlined Docker setup.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and `uv`.
2.  Clone the repository and sync dependencies:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env
    ```

3.  Populate the `.env` file with necessary API keys, following the examples provided:

    ```bash
    #  LLM configuration (example using DeepSeek)
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    ```bash
    # (Free tokens for DeepSeek - Tencent Cloud International)
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

See [`docker/README.md`](./docker/README.md).

### Quick Start

Run a default agent with a search tool:

```bash
# Requires SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --stream --config default

# Without the search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

*   Configure tool APIs in `.env` (e.g., `SERPER_API_KEY`, `JINA_API_KEY`).

To generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

For a web UI visualization, download and install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```

Run web version of the SVG image generation command:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link:

```
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on `WebWalkerQA`:

```bash
# Prepare dataset.
python scripts/data/process_web_walker_qa.py

# Run evaluation. Set required values in .env.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

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