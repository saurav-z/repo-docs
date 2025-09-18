<!-- Improved & Summarized README.md -->

# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent empowers you to create, deploy, and evaluate autonomous agents effortlessly, leveraging the power of open-source models.**  [Explore the Youtu-Agent Repository](https://github.com/Tencent/Youtu-agent)

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/Tencent/Youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

---

## Key Features

*   ✅ **Open-Source Focused:** Designed for cost-effective deployment using accessible open-source models.
*   🚀 **High-Performance Benchmarks:** Achieved leading results on WebWalkerQA and GAIA benchmarks, demonstrating strong capabilities.
*   ⚙️ **Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), supporting various model APIs and tool integrations.
*   💡 **Practical Use Cases:** Includes out-of-the-box support for data analysis, file processing, and content generation (e.g., CSV analysis, research reports, and video generation).
*   🤖 **Automated Agent Generation:** Streamlined YAML-based configuration and automatic agent generation simplifies setup and reduces manual effort.

---

## What is Youtu-Agent?

Youtu-Agent is a cutting-edge framework for building and deploying intelligent agents that can perform complex tasks using open-source models. It offers a flexible, high-performance architecture optimized for practical applications.

---

## Key Highlights:

*   **Strong Performance:**  Achieved impressive results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8% on the text-only subset), proving its effectiveness with models like DeepSeek-V3.
*   **Cost-Effective & Open-Source:**  Focuses on leveraging open-source models, reducing costs and increasing accessibility.
*   **Real-World Applications:**  Ready-to-use examples for tasks like data analysis, file management, research reports, and more.
*   **Modular and Extensible:** The framework's design makes it easy to integrate new models, tools, and functionalities.
*   **Simplified Setup:** YAML-based configuration and automated agent generation significantly speed up development.

---

## 📢 News

*   **[2025-09-09]**  Design philosophy and basic usage live stream: [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)]
*   **[2025-09-02]**  Free tokens for DeepSeek API users (Tencent Cloud International):  New users can get **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381).
*   **[2025-08-28]**  DeepSeek-V3.1 updates live stream: [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)]

---

## 🌟 Benchmark Performance

Youtu-Agent demonstrates strong results on deep search and tool use benchmarks:

*   **WebWalkerQA:** Achieved 71.47% with DeepSeek-V3.1.
*   **GAIA (text-only subset):**  Achieved 72.8% pass@1 using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

---

## 💡 Examples

Click on the images to view detailed videos:

| **Data Analysis**                                                     | **File Management**                                                   |
| :-------------------------------------------------------------------- | :------------------------------------------------------------------- |
| Analyzes a CSV file and generates an HTML report.                     | Renames and categorizes local files for the user.                    |
| <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920"  poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video>  |
| **Wide Research**                                                     | **Paper Analysis**                                                    |
| Gathers extensive information to generate a comprehensive report.      | Parses a paper, performs analysis, and compiles related literature. |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d"  poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |

>   [!NOTE]
>   See the  [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

**Youtu-Agent's unique auto-generation feature simplifies agent creation.** Define agent tasks using simple YAML configs, eliminating the need for complex prompt engineering.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

| **Automatic Agent Generation**                                                  |
| :----------------------------------------------------------------------------- |
| Interactively clarify requirements, automatically generate the configuration. |
| <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" controls muted preload="metadata" width="100%" height="auto" style="object-fit: cover; border-radius: 8px;"></video> |

>   [!NOTE]
>   See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

---

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Simple and easy to use, avoiding unnecessary complexity.
*   **Modular & Configurable:**  Flexible customization and integration of new components.
*   **Open-Source Model Support & Low-Cost:** Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation

*   **YAML based configuration:** Structured and easily manageable agent configurations.
*   **Automatic agent generation:** Based on user requirements, agent configurations can be automatically generated.
*   **Tool generation & optimization:** Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide research:** Covers common search-oriented tasks.
*   **Webpage generation:** Examples include generating web pages based on specific inputs.
*   **Trajectory collection:** Supports data collection for training and research purposes.

---

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent benefits:

### For Agents Researchers & LLM Trainers
- A **simple yet powerful baseline** that is stronger than basic ReAct, serving as an excellent starting point for model training and ablation studies.
- **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers
- A **proven and portable scaffolding** for building real-world agent applications.
- **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
- **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts
- **Practical Use Cases**: The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
- **Simplicity & Debuggability**: A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

---

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools.
*   **Environment:** The world the agent operates in (e.g., a browser).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Encapsulated workflow for specific datasets.

For more details, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

---

## 🚀 Getting Started

Follow these steps to run your first agent, or refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Setup

#### Source Code Deployment

>   [!NOTE]
>   Requires Python 3.12+.  Recommend [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```

3.  Install dependencies:

    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```

4.  Configure your environment:

    ```bash
    cp .env.example .env  # Fill in your API keys.
    ```

    Example `.env` (replace placeholders):

    ```bash
    # LLM (DeepSeek example, or configure your preferred LLM)
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    or using Tencent Cloud:

    ```bash
    # LLM
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup with an interactive frontend.

### Quick Start

1.  Run a simple agent:

    ```bash
    # NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
    # (We plan to replace these with free alternatives in the future)
    python scripts/cli_chat.py --stream --config default
    # To avoid using the search toolkit, you can run:
    python scripts/cli_chat.py --stream --config base
    ```

    📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure API keys in `.env` for the tools module to use web search.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image on the topic of “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

Web UI for the SVG example:

```bash
# Download and install the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
python examples/svg_generator/main_web.py
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

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

---

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide to get you up and running.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions and issues.

---

## 🙏 Acknowledgements

This project is built on the shoulders of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

---

## 🙌 Contributing

We welcome contributions! Read our [**Contributing Guidelines**](./CONTRIBUTING.md).

---

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

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)](https://star-history.com/#TencentCloudADP/youtu-agent)