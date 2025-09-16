# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework that empowers you to build, run, and evaluate autonomous agents, achieving impressive results with open-source models. Explore the framework on its [GitHub Repository](https://github.com/Tencent/Youtu-agent).

**Key Features:**

*   ✅ **High Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA and GAIA using open-source models (e.g., DeepSeek-V3 series)
*   💰 **Cost-Effective:** Designed for low-cost deployment and open-source model compatibility.
*   💡 **Practical Use Cases:** Supports data analysis, file processing, literature review, and more.
*   ⚙️ **Flexible Architecture:** Built on openai-agents with extensive support for diverse models, tools, and frameworks.
*   🚀 **Automation & Simplicity:** Streamlined agent creation with YAML-based configuration and automatic generation.

## Table of Contents

*   [✨ Features](#-features)
*   [🌟 Benchmark Performance](#-benchmark-performance)
*   [💡 Examples](#-examples)
    *   [🤖 Automatic Agent Generation](#-automatic-agent-generation)
*   [🤔 Why Choose Youtu-Agent?](#-why-choose-youtu-agent)
*   [🚀 Getting Started](#-getting-started)
    *   [Setup](#setup)
    *   [Quick Start](#quick-start)
    *   [Explore More Examples](#explore-more-examples)
    *   [Run Evaluations](#run-evaluations)
*   [📖 Dive Deeper](#-dive-deeper)
*   [🙏 Acknowledgements](#-acknowledgements)
*   [🙌 Contributing](#-contributing)
*   [📚 Citation](#-citation)
*   [⭐ Star History](#-star-history)

## ✨ Features

Youtu-Agent offers a modular design that prioritizes simplicity and adaptability.  Its core features include:

*   **Built on openai-agents:** Leverages the openai-agents SDK for seamless integration with various LLM APIs and features like streaming and tracing.
*   **Fully asynchronous:** Enables high-performance, efficient execution, especially beneficial for benchmarking.
*   **Tracing & analysis system:** Provides in-depth analysis of tool calls and agent trajectories. (Coming soon)
*   **YAML based configuration:** Structured and easily manageable agent configurations.
*   **Automatic agent generation:** Generate agent configurations based on user requirements automatically.
*   **Tool generation & optimization:** Planned for future support of tool evaluation, automated optimization, and customized tool generation.

### Design Philosophy

*   **Minimal design:** Keeps the framework simple and easy to use, avoiding unnecessary overhead.
*   **Modular & configurable:** Flexible customization and easy integration of new components.
*   **Open-source model support & low-cost:** Promotes accessibility and cost-effectiveness for various applications.

## 🌟 Benchmark Performance

Youtu-Agent excels in benchmark performance, showcasing its capabilities using open-source models and lightweight tools:

*   **WebWalkerQA:** Achieved 71.47% accuracy with DeepSeek-V3.1.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Youtu-Agent provides several ready-to-use examples demonstrating its versatility:

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
      <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920" 
             poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" 
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
      <strong>Paper Analysis</strong><br>Parses a given paper and compiles related literature.
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

> [!NOTE]
> See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with automatic configuration generation. Define your agent's requirements, and the framework will generate and save the configuration automatically.

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

> [!NOTE]
> See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides significant value to different user groups:

*   **For Agents Researchers & LLM Trainers:** Serves as a strong baseline for model training and ablation studies.
*   **For Agent Application Developers:** Provides a proven scaffolding for building real-world agent applications with ease of use and modular design.
*   **For AI & Agent Enthusiasts:** Offers practical use cases and simplicity for development and debugging.

## 🚀 Getting Started

Get up and running quickly with the following setup and quickstart guide.

### Setup

**Prerequisites:** Python 3.12+ and [uv](https://github.com/astral-sh/uv) (recommended for dependency management).

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```

2.  **Install dependencies:**

    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```

3.  **Configure Environment Variables:**

    ```bash
    cp .env.example .env  # Create a .env file and fill in your API keys
    ```

    *   **DeepSeek API (Recommended):**

        ```bash
        # llm
        # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
        UTU_LLM_TYPE=chat.completions
        UTU_LLM_MODEL=deepseek-v3
        UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
        UTU_LLM_API_KEY=replace-with-your-api-key
        ```

    *   **OpenAI API (if using OpenAI-compatible models):**

        ```bash
        # llm requires OpenAI API format compatibility
        # setup your LLM config , ref https://api-docs.deepseek.com/
        UTU_LLM_TYPE=chat.completions
        UTU_LLM_MODEL=deepseek-chat
        UTU_LLM_BASE_URL=https://api.deepseek.com/v1
        UTU_LLM_API_KEY=replace-to-your-api-key
        ```

    *   **Tools (Search, etc.):** Configure API keys in `.env` as needed.

        ```bash
        # tools
        # serper api key, ref https://serper.dev/playground
        SERPER_API_KEY=<Your Serper API Key>
        # jina api key, ref https://jina.ai/reader
        JINA_API_KEY=<Your Jina API Key>
        ```

### Quick Start

Run a simple agent using a built-in configuration:

```bash
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Explore and run the provided examples:

```bash
python examples/svg_generator/main.py
```

For the web UI, download the frontend package, install it, and then run:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

python examples/svg_generator/main_web.py
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking on standard datasets:

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

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Refer to the [**Contributing Guidelines**](./CONTRIBUTING.md) for details.

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