# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework, empowering developers to create, run, and evaluate intelligent agents using open-source models. **Unlock the potential of AI agents with Youtu-Agent, achieving state-of-the-art results on benchmarks while offering cost-effective deployment.**  ([Original Repository](https://github.com/Tencent/Youtu-agent))

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#-key-features"><b>✨ Key Features</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#-examples"><b>💡 Examples</b> </a>
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

## ✨ Key Features

*   **Top-tier Performance**: Achieved impressive results on WebWalkerQA and GAIA benchmarks using open-source `DeepSeek-V3` models, showcasing strong capabilities.
*   **Open-Source & Cost-Effective**: Designed for accessible and affordable agent development, minimizing reliance on proprietary models.
*   **Practical Use Cases**: Supports various tasks like CSV analysis, research, file management, and upcoming features for podcast/video generation.
*   **Flexible Architecture**: Built on the robust [openai-agents](https://github.com/openai/openai-agents-python) framework, supporting diverse model APIs, tool integrations, and framework implementations.
*   **Streamlined Automation**: YAML-based configuration, automatic agent generation, and simplified setup to reduce manual configuration and speed up your workflow.

## 🗞️ News

*   **[2025-09-09]**: Hosted a live sharing the design philosophy and basic usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **[2025-09-02]**: [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]**: Hosted a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent excels in benchmarks using open-source models and lightweight tools, demonstrating strong performance in challenging deep search and tool use tasks.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 71.47% accuracy with `DeepSeek-V3.1`, establishing a new SOTA.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click on the images to view detailed videos.

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

> [!NOTE]
> See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent features automatic agent generation using simple YAML-based configurations, streamlining agent creation. The "meta-agent" interacts with you to capture requirements and automatically creates the configuration.

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

## ✨ Features

### Design Philosophy

*   **Minimal Design**: Simple and easy-to-use framework.
*   **Modular & Configurable**: Offers flexible customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost**: Focuses on accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents**: Leverages the [openai-agents](https://github.com/openai/openai-agents-python) SDK for features like streaming, tracing, and agent-loop capabilities, ensuring compatibility with diverse models.
*   **Fully Asynchronous**: Enables high-performance and efficient execution.
*   **Tracing & Analysis System**: Provides in-depth analysis of tool calls and agent trajectories using the `DBTracingProcessor` system. (will be released soon)

### Automation

*   **YAML-based Configuration**: Uses structured and easily manageable agent configurations.
*   **Automatic Agent Generation**: Generates agent configurations based on user requirements.
*   **Tool Generation & Optimization**: Supports tool evaluation, automated optimization, and customized tool generation (future).

### Use Cases

*   **Deep / Wide Research**: Addresses common search-oriented tasks.
*   **Webpage Generation**: Examples include generating web pages based on specific inputs.
*   **Trajectory Collection**: Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed for:

### Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline**, serving as an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** to streamline experimentation.

### Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Quick start with simple scripts and built-in toolkits.
*   **Modular Design**: Customizable components like `Environment` and `ContextManager`.

### AI & Agent Enthusiasts

*   **Practical Use Cases**: Examples include deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability**: A rich toolset and visual tracing tools.

## 🧩 Core Concepts

*   **Agent**: An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit**: An encapsulated set of tools that an agent can use.
*   **Environment**: The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager**: A configurable module for managing the agent's context window.
*   **Benchmark**: An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more details, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to run your first agent:

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository:
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```
3.  Sync dependencies:
    ```bash
    uv sync  # or, `make sync`
    ```
4.  Activate the virtual environment:
    ```bash
    source ./.venv/bin/activate
    ```
5.  Configure the `.env` file with your API keys:
    ```bash
    cp .env.example .env
    ```
    Populate the `.env` file with necessary API keys for your LLM and tools. For example:

    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    Or use the [Tencent Cloud International](https://www.tencentcloud.com/) DeepSeek API free tokens:

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with an interactive frontend.

### Quick Start

Run a simple agent using the `configs/agents/simple/base_search.yaml` configuration:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env`:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Automatically search and generate an SVG image:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime using the web UI, download the frontend package and install it locally:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run the web version:

```bash
python examples/svg_generator/main_web.py
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

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

Results are stored and can be analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Answers to common questions.

## 🙏 Acknowledgements

-   [openai-agents](https://github.com/openai/openai-agents-python)
-   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
-   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

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