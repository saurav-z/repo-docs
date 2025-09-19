# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a versatile framework empowering developers to create, deploy, and evaluate intelligent agents, leveraging the power of open-source models.  [Explore the Youtu-Agent Repository](https://github.com/Tencent/Youtu-agent) for cutting-edge agent development!

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
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

Youtu-Agent provides a flexible, high-performance framework for constructing, executing, and assessing autonomous agents. It offers advanced agent capabilities, including data analysis, file processing, and in-depth research, using open-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   **High-Performance Benchmarks:** Achieved impressive results on WebWalkerQA and GAIA benchmarks using exclusively DeepSeek-V3 series models.
*   **Cost-Effective Open-Source Focus:** Designed for accessible and low-cost deployment without relying on proprietary models.
*   **Practical Application Support:** Out-of-the-box solutions for diverse tasks like CSV analysis, literature reviews, and file management.
*   **Flexible & Extensible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), with comprehensive support for various model APIs and tool integrations.
*   **Automated Agent Generation:** YAML-based configuration, auto-agent creation, and streamlined setup.

## 🗞️ News

*   **(2025-09-09)**: Live sharing of design philosophy and usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)]
*   **(2025-09-02)**: [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) with the DeepSeek models in `Youtu-Agent`. For enterprise agent solutions, check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **(2025-08-28)**: Live sharing of updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)]

## 🌟 Benchmark Performance

Youtu-Agent utilizes open-source models and lightweight tools, delivering strong results on challenging deep search and tool-use benchmarks.

*   **WebWalkerQA:** Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new SOTA.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.  Multimodal tool evaluation is in progress.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images to view detailed video demonstrations.

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
            <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report, replicating the functionality of Manus.
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
                   width="100%" height="300"
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

>   [!NOTE]
>   See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its **automatic agent generation** capability. Simply define your requirements, and the framework handles the rest.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Automatic Agent Generation</strong><br>Interactively capture requirements and automatically generate and run the agent configuration.
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

>   [!NOTE]
>   See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal design**: Simplifies the framework for ease of use.
*   **Modular & configurable**: Enables flexible customization and easy integration.
*   **Open-source model support & low-cost**: Provides accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents**:  Leverages the openai-agents SDK for streaming, tracing, and agent-loop capabilities, ensuring broad model compatibility.
*   **Fully asynchronous**:  Enables high-performance execution, particularly for evaluations.
*   **Tracing & analysis system**:  Offers in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation

*   **YAML based configuration**: Structured and easily manageable agent configurations.
*   **Automatic agent generation**: Agent configuration is generated based on user requirements.
*   **Tool generation & optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide research**: Supports common search-oriented tasks.
*   **Webpage generation**: Includes generating web pages based on specific inputs.
*   **Trajectory collection**: Enables data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides value for different user groups:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline**, ideal for model training and ablation studies.
*   **One-click evaluation scripts** streamline the experimental process.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and toolkits.
*   **Modular Design**: Encapsulated and customizable components.

### For AI & Agent Enthusiasts

*   **Practical Use Cases**: Explore diverse tasks.
*   **Simplicity & Debuggability**: A rich toolset facilitates development and debugging.

## 🧩 Core Concepts

*   **Agent**: An LLM with specific prompts, tools, and an environment.
*   **Toolkit**: A set of tools that an agent can use.
*   **Environment**: The world the agent operates in.
*   **ContextManager**: Manages the agent's context window.
*   **Benchmark**: An encapsulated workflow for a dataset.

For details, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides code and examples to help you get started.

### Setup

#### Source Code Deployment

>   [!NOTE]
>   Requires Python 3.12+. Use [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: Configure the necessary API keys.
```

Fill in the required keys in `.env`, e.g., your LLM API key:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

>   [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace the API key in the .env file:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

See [`docker/README.md`](./docker/README.md).

### Quick Start

Use `configs/agents/simple/base_search.yaml` for a search-enabled agent:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run the CLI chatbot:

```bash
# NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
python scripts/cli_chat.py --stream --config simple/base_search
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` for search-enabled examples:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

Web UI (download from Youtu-Agent releases):

```bash
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
python examples/svg_generator/main_web.py
```

Access the project at `http://127.0.0.1:8848/`.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on datasets like `WebWalkerQA`:

```bash
python scripts/data/process_web_walker_qa.py
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

-   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts.
-   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get started quickly.
-   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Read our [**Contributing Guidelines**](./CONTRIBUTING.md).

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