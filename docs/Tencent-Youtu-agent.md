# Youtu-Agent: Build Powerful Agents with Open-Source Models

> Unleash the power of open-source models with Youtu-Agent, a flexible framework for building, running, and evaluating autonomous agents.

[<img src="https://img.shields.io/badge/GitHub-Tencent-blue.svg" alt="GitHub">](https://github.com/Tencent/Youtu-agent)
[<img src="https://img.shields.io/badge/📖-Documentation-blue.svg" alt="Documentation">](https://tencentcloudadp.github.io/youtu-agent/)

Youtu-Agent empowers developers to create sophisticated autonomous agents capable of data analysis, file processing, and deep research, all leveraging open-source models. This framework offers high performance, cost-effectiveness, and ease of use.

**Key Features:**

*   ✅ **High Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8% on text-only subset) using open-source DeepSeek-V3 models.
*   💰 **Cost-Effective:** Designed for accessible and low-cost deployment, avoiding reliance on expensive closed-source models.
*   🛠️ **Practical Use Cases:** Out-of-the-box support for tasks like CSV analysis, literature review, file organization, and more, with podcast/video generation coming soon.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting diverse model APIs, tool integrations, and framework implementations.
*   🤖 **Automation & Simplicity:** YAML-based configs and automatic agent generation reduce manual effort.

## ✨ Core Features & Benefits

*   **Minimal Design:** Simple and user-friendly framework to avoid unnecessary overhead.
*   **Modular & Configurable:** Offers flexible customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost:** Promotes accessibility and cost-effectiveness for various applications.
*   **Built on openai-agents:** Inherits streaming, tracing, and agent-loop capabilities.
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories.
*   **YAML-based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generates agent configurations based on user requirements.
*   **Tool Generation & Optimization:** Provides tool evaluation and automated optimization, with future support for customized tool generation.

## 🚀 Getting Started

### 1. Setup

**Prerequisites:** Python 3.12+ is required. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure your API keys!
```

Populate your `.env` file with necessary API keys (e.g., LLM API keys). Example configuration:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

**Free Tokens:** New users of the DeepSeek API get **3 million free tokens** (**Sep 1 – Oct 31, 2025**). Replace the API key in your `.env` file after applying for free tokens.

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

### 2. Quick Start

Run a simple agent with a search tool using:

```bash
# Configure SERPER_API_KEY and JINA_API_KEY in `.env` for web search.
python scripts/cli_chat.py --config simple/base_search
```

Or, run without the search tool:

```bash
python scripts/cli_chat.py --config simple/base
```

📖 Learn more: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### 3. Explore Examples

Enable the agent to automatically search and generate an SVG image:

```bash
python examples/svg_generator/main.py
```

For a web UI, install the frontend package and run:

```bash
# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

# Run the web version
python examples/svg_generator/main_web.py
```

Access the project via the displayed local link.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will then search and generate an SVG visualization:

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### 4. Run Evaluations

Evaluate on WebWalkerQA:

```bash
# Prepare dataset.
python scripts/data/process_web_walker_qa.py

# Run evaluation (adjust parameters as needed):
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🤔 Why Choose Youtu-Agent?

*   **For Researchers:** A strong baseline for model training and ablation studies.
*   **For Developers:** A proven framework for real-world agent applications.
*   **For Enthusiasts:** Practical use cases and easy debugging.

## 🧩 Core Concepts

*   **Agent:** LLM with prompts, tools, and an environment.
*   **Toolkit:** Encapsulated tools for the agent.
*   **Environment:** The agent's operating world (e.g., browser).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Encapsulated workflow for datasets.

Explore the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for more details.

## 🙌 Contributing

We welcome your contributions! See our [**Contributing Guidelines**](./CONTRIBUTING.md).

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