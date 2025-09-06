<!-- ALL-IN-ONE SEO-OPTIMIZED README -->
# Youtu-Agent: Unleash the Power of Open-Source Agents with Ease

**Youtu-Agent is a cutting-edge, open-source agent framework designed to build, run, and evaluate autonomous agents with exceptional performance and cost-effectiveness. Check out the original repo: [https://github.com/TencentCloudADP/youtu-agent](https://github.com/TencentCloudADP/youtu-agent)**

<div align="center">
    <a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
    <a href="https://github.com/TencentCloudADP/youtu-agent"><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
    <a href="https://deepwiki.com/TencentCloudADP/youtu-agent"><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
    <a href="README_ZH.md"><b>中文版</b></a> |
    <a href="#benchmark-performance"><b>🌟 Performance</b></a> |
    <a href="#examples"><b>💡 Examples</b> </a> |
    <a href="#features"><b>✨ Features</b> </a> |
    <a href="#getting-started"><b>🚀 Getting Started</b> </a> |
    <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Mascot" width="200" align="left" style="margin-right:20px;">

Youtu-Agent empowers you to build intelligent agents for a variety of tasks, leveraging the power of open-source models. Achieve impressive results in data analysis, file processing, and research with a simple and powerful framework.

**Key Highlights:**

*   **Superior Performance:** Achieved state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%, text-only), using the open-source DeepSeek-V3 series models.
*   **Cost-Effective & Open-Source:** Optimized for accessible, low-cost deployments, eliminating dependence on proprietary models.
*   **Practical Use Cases:** Supports tasks like CSV analysis, literature reviews, personal file organization, and more.  Video and podcast generation are coming soon.
*   **Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), offering extensible support for diverse model APIs, tool integrations, and framework implementations.
*   **Automated & Simplified:** YAML-based configurations, automatic agent generation, and streamlined setup to reduce manual effort.

## 🗞️ News

*   **[2025-09-02] DeepSeek API Free Tokens:** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381) to use DeepSeek models in Youtu-Agent! For enterprise agent solutions, explore the [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28] DeepSeek-V3.1 and Youtu-Agent Updates:**  Watch the live sharing updates about DeepSeek-V3.1 and its integration with the `Youtu-Agent` framework.  [Documentation Link](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent delivers strong results on challenging deep search and tool use benchmarks, all powered by open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA):** Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/):** Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`. Evaluation on the full GAIA benchmark is in progress.

![WebWalkerQA Benchmark](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the power of Youtu-Agent with these interactive examples.  Click the images to watch detailed video demonstrations.

<table style="width:100%;">
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Data Analysis</strong><br>Analyze CSV data and generate an HTML report.
        </td>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>File Management</strong><br>Rename and categorize local files.
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
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Wide Research</strong><br>Generate comprehensive reports based on extensive research.
        </td>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Paper Analysis</strong><br>Analyze research papers and compile related literature.
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

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its automatic agent generation capabilities.  Define agents with easy-to-use YAML-based configurations, eliminating the need for extensive coding or prompt engineering.

```bash
# Interactively define requirements and generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Automatic Agent Generation</strong><br>Easily define agent requirements and auto-generate the agent configuration for instant execution.
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid black; padding: 10px; vertical-align:top; width: 400px;">
            <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef"
                   poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg"
                   controls muted preload="metadata"
                   width="100%" height="auto"
                   style="object-fit: cover; border-radius: 8px;"></video>
        </td>
    </tr>
</table>

Explore more examples and advanced use-cases in the [`examples`](./examples) directory and the comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Simple and easy-to-use framework to avoid unnecessary overhead.
*   **Modular & Configurable:** Flexible customization and easy integration of new components.
*   **Open-Source & Low-Cost:** Promotes accessibility and cost-effectiveness for diverse applications.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python), ensuring compatibility with both `responses` and `chat.completions` APIs for diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution, especially for benchmark evaluation.
*   **Tracing & Analysis System:**  A `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (Coming soon!)

### Automation

*   **YAML-Based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generate agent configurations based on user requirements.
*   **Tool Generation & Optimization:** Tool evaluation, automated optimization, and customized tool generation are planned for the future.

### Use Cases

*   **Deep/Wide Research:** Supports common search-oriented tasks.
*   **Webpage Generation:** Includes examples of generating web pages from specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed to provide value to various user groups:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** that is stronger than basic ReAct, providing an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:** Quickly get started with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The context in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For details about design and implementation, refer to the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides complete code and examples to help you get started quickly. Follow the steps below to run your first agent, or refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with an interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Install Prerequisites:** Ensure Python and `uv` are installed.
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```
3.  **Sync Dependencies:**
    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```
4.  **Configure Environment Variables:**
    ```bash
    cp .env.example .env  # Copy the example file
    ```
    Fill in the necessary API keys in the `.env` file, such as the LLM API keys:
    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```
    >   Take advantage of the [Tencent Cloud International](https://www.tencentcloud.com/) offer: new DeepSeek API users receive **3 million free tokens** (**Sep 1 – Oct 31, 2025**). After applying, update the `.env` file:
    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

For a streamlined Docker-based setup with an interactive frontend, see [`docker/README.md`](./docker/README.md).

### Quick Start

Youtu-agent includes pre-built configurations.  The default config (`configs/agents/default.yaml`) defines a simple agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Launch an interactive CLI chatbot with this agent using:

```bash
# Requires `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search. (Will be replaced with free alternatives later)
python scripts/cli_chat.py --stream --config default
# Without the search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

The repository has ready-to-use examples.  Some require internet search, so configure tool APIs in the `.env` file:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To enable the agent to search online and generate an SVG image, run:

```bash
python examples/svg_generator/main.py
```

To visualize the agent’s runtime with the web UI, download and install the frontend package:

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

Access the project via the local link after the terminal displays:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent searches the web, gathers information, and outputs an SVG visualization based on a research topic.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking on standard datasets. To evaluate on `WebWalkerQA`:

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation
# Set `JUDGE_LLM_TYPE`, etc. in `.env`.  Use `WebWalkerQA_15` for quick evaluation.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project builds on the contributions of these open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

If you use this project, please cite it:

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

[![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)](https://star-history.com/#TencentCloudADP/youtu-agent)