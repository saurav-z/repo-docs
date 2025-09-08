# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

Youtu-Agent empowers developers to create advanced AI agents, offering robust performance and cost-effective deployment using open-source models. [Explore the Youtu-Agent repository](https://github.com/TencentCloudADP/Youtu-agent) for cutting-edge agent development.

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

**Key Features:**

*   **High-Performance & Open-Source:** Achieve state-of-the-art results on benchmarks like WebWalkerQA and GAIA using open-source models like DeepSeek-V3, ensuring accessible and cost-effective agent development.
*   **Practical Use Cases:** Supports various tasks out-of-the-box, including CSV analysis, literature review, file management, and more.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), offering extensibility for diverse model APIs, tool integrations, and framework implementations.
*   **Automated Agent Generation:** Simplify agent creation with YAML-based configurations and automatic agent generation, minimizing manual setup.
*   **Comprehensive Benchmarking:** Evaluate agent performance with streamlined evaluation scripts, facilitating reproducible research and development.

## Table of Contents

*   [🌟 Benchmark Performance](#-benchmark-performance)
*   [💡 Examples](#-examples)
*   [🤖 Automatic Agent Generation](#-automatic-agent-generation)
*   [✨ Features](#-features)
*   [🤔 Why Choose Youtu-Agent?](#-why-choose-youtu-agent)
*   [🚀 Getting Started](#-getting-started)
*   [🙏 Acknowledgements](#-acknowledgements)
*   [📚 Citation](#-citation)

## 🌟 Benchmark Performance

Youtu-Agent excels on challenging benchmarks, demonstrating the power of open-source models.

*   **WebWalkerQA:** Achieved 71.47% accuracy (pass@1) with DeepSeek-V3.1, setting a new state-of-the-art.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore Youtu-Agent's capabilities with these interactive examples:

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
            <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report.
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

## 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation.

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

## ✨ Features

*   **Minimal Design:** Simple and easy to use.
*   **Modular & Configurable:** Flexible customization and integration of new components.
*   **Open-Source Model Support & Low-Cost:** Accessible and cost-effective.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories (will be released soon).

### Automation

*   **YAML-based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generates configurations based on user requirements.
*   **Tool Generation & Optimization:** Tool evaluation and automated optimization (future support).

### Use Cases

*   Deep / Wide Research
*   Webpage Generation
*   Trajectory Collection

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant advantages for various users:

### For Agents Researchers & LLM Trainers
*   **Simple Baseline:** A strong starting point for model training and ablation studies.
*   **One-Click Evaluation:** Streamlined benchmarking.

### For Agent Application Developers
*   **Proven Scaffolding:** For building real-world agent applications.
*   **Ease of Use:** Simple scripts and built-in toolkits.
*   **Modular Design:** Customizable components.

### For AI & Agent Enthusiasts
*   **Practical Use Cases:** Examples for deep research, data analysis, and file organization.
*   **Simplicity & Debuggability:** Rich toolset and visual tracing tools.

## 🚀 Getting Started

Follow these steps to set up and run your first agent:

### Setup

1.  **Install Dependencies:**

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Config API keys here.
    ```
2.  **Configure API Keys:** Fill in the necessary API keys in the `.env` file.
    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```
    Or use the free tokens from Tencent Cloud International, if you are a new user:
    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

### Quick Start
Run a CLI chatbot with a search tool:
```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples
Enable the agent to search the web and generate an SVG image:
```bash
python examples/svg_generator/main.py
```
Run the web version:
```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```
Next, run the web version of the SVG image generation command:
```bash
python examples/svg_generator/main_web.py
```
📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations
Evaluate on `WebWalkerQA`:
```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```
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