# Youtu-Agent: Build Powerful Agents with Open-Source Models

**[Explore Youtu-Agent on GitHub](https://github.com/TencentCloudADP/Youtu-agent)**

Youtu-Agent is a cutting-edge agent framework enabling you to build, run, and evaluate autonomous agents, all leveraging the power of open-source models.

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

---

**Key Features:**

*   🚀 **Performance Leader:** Achieves impressive results, including 71.47% on WebWalkerQA and 72.8% on GAIA (text subset), demonstrating strong capabilities with open-source models.
*   💡 **Open-Source & Cost-Effective:** Designed for accessible and affordable deployment, avoiding reliance on proprietary models.
*   ✨ **Practical Use Cases:** Supports various tasks like CSV analysis, research, file organization, and media generation (coming soon).
*   🛠️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), offers broad model and tool integration.
*   ⚙️ **Simplified Automation:** YAML-based configuration, auto-agent generation, and streamlined setup reduce complexity.

---

## 🚀 Getting Started

Jumpstart your agent development with Youtu-Agent. Follow these simple steps:

### 1. Setup
  #### Source Code Deployment
    1. Ensure Python 3.12+ and `uv` (recommended) are installed.
    2. Clone the repository and set up dependencies:
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure your API keys here.
    ```
    3. Populate the `.env` file with your LLM API keys (e.g., DeepSeek, OpenAI).
  #### Docker Deployment
  Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based deployment.

### 2. Quick Start
  1. Run an interactive CLI chatbot with a search tool:
    ```bash
    # Configure API keys for web search in `.env` (SERPER_API_KEY, JINA_API_KEY)
    python scripts/cli_chat.py --stream --config default
    ```
  2. Launch a simple agent (without search) if you skip the tool configuration:
    ```bash
    python scripts/cli_chat.py --stream --config base
    ```

### 3. Explore Examples
  * Enable search tools in `.env` (SERPER_API_KEY, JINA_API_KEY) to run examples.
  * Run SVG image generation:
    ```bash
    python examples/svg_generator/main.py
    ```
  *  View the web UI and visualization by running:
    ```bash
    uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
    python examples/svg_generator/main_web.py
    ```
    *Access the project by the local link shown by terminal*

### 4. Run Evaluations
  * Evaluate on WebWalkerQA:
    ```bash
    python scripts/data/process_web_walker_qa.py
    python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
    ```
  * Review the results using the evaluation platform for analysis ([Evaluation Analysis](./frontend/exp_analysis/README.md)).

---

## 🌟 Key Highlights & Benefits

*   **Performance:** Demonstrated strong results on WebWalkerQA (71.47%) and GAIA (72.8%) using open-source models, making it a strong baseline for agent development.
*   **Open-Source Focus:** Cost-effective and accessible, enabling users to build powerful agents without relying on closed-source models.
*   **Versatile Use Cases:** Supports diverse applications including data analysis, research, and content generation.
*   **Flexible & Extensible:** Built on [openai-agents](https://github.com/openai/openai-agents-python), allowing easy integration of various models, tools, and frameworks.
*   **Simplified Development:** YAML configurations and automatic agent generation reduce development time and effort.

---

## ✨ Features

### Core Features
- **Built on openai-agents**: Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
- **Fully asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
- **Tracing & analysis system**: Beyond OTEL, our `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
- **YAML based configuration**: Structured and easily manageable agent configurations.
- **Automatic agent generation**: Based on user requirements, agent configurations can be automatically generated.
- **Tool generation & optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases
- **Deep / Wide research**: Covers common search-oriented tasks.
- **Webpage generation**: Examples include generating web pages based on specific inputs.
- **Trajectory collection**: Supports data collection for training and research purposes.


## 🤔 Why Choose Youtu-Agent?

`Youtu-Agent` is designed to provide significant value to different user groups:

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


## 🧩 Core Concepts

- **Agent**: An LLM configured with specific prompts, tools, and an environment.
- **Toolkit**: An encapsulated set of tools that an agent can use.
- **Environment**: The world in which the agent operates (e.g., a browser, a shell).
- **ContextManager**: A configurable module for managing the agent's context window.
- **Benchmark**: An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

---

## 🗞️ News

*   **[2025-09-02]** DeepSeek API Free Tokens: [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381). For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]** Live Sharing Updates: Updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [Documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

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

## 🙏 Acknowledgements

This project builds upon the excellent work of several open-source projects:
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

---

## 🌟 Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)