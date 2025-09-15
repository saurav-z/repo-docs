# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Unlock the power of autonomous agents with Youtu-Agent, a flexible and high-performing framework, now with a link to the original repository:  [https://github.com/TencentCloudADP/Youtu-agent](https://github.com/TencentCloudADP/Youtu-agent).**

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a> 
| <a href="#-examples"><b>💡 Examples</b> </a> 
| <a href="#-features"><b>✨ Features</b> </a> 
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a> 
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a> 
</p>

Youtu-Agent empowers you to build, run, and evaluate AI agents capable of complex tasks, leveraging the efficiency and cost-effectiveness of open-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   🚀 **High Performance:** Achieves impressive benchmark results using open-source models, including 71.47% on WebWalkerQA (pass@1) and 72.8% on GAIA (text-only subset, pass@1) with DeepSeek-V3 series.
*   💰 **Cost-Effective:** Designed for accessible, low-cost deployment, eliminating reliance on expensive closed models.
*   💡 **Practical Use Cases:** Supports diverse tasks such as CSV analysis, literature reviews, file organization, and soon, podcast/video generation.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), allowing extensibility with different model APIs, tool integrations, and framework implementations.
*   🤖 **Automated Configuration:** YAML-based configs, auto-agent generation, and simplified setup to reduce manual effort.

## 📰 News

*   🎁 \[2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 \[2025-08-28] We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels in challenging deep search and tool use benchmarks due to its reliance on open-source models and lightweight tools.

*   **WebWalkerQA:**  Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.  Multimodal tool evaluation is in progress.

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

### 🤖 Automatic Agent Generation

Youtu-Agent streamlines agent creation with its automatic agent generation feature. Define requirements in simple YAML configs, and the "meta-agent" will handle the rest.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define needs, automatically generate configurations, and run agents effortlessly.
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


Refer to the [`examples`](./examples) directory and [`docs/examples.md`](./docs/examples.md) for more detailed examples and advanced use cases.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
-   **Minimal Design**: Simple and easy to use, with minimal overhead.
-   **Modular & Configurable**: Flexible customization and easy integration of new components.
-   **Open-Source & Low-Cost**: Promotes accessibility and cost-effectiveness.

### Core Features
-   **Built on openai-agents**: Leverages the robust foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK.
-   **Fully Asynchronous**: Enables high-performance execution.
-   **Tracing & Analysis System**:  In-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
-   **YAML Configuration**:  Manages agent configurations in a structured manner.
-   **Automatic Agent Generation**:  Automates configuration creation based on user input.
-   **Tool Generation & Optimization**: Future support for tool evaluation, optimization, and custom tool creation.

### Use Cases
-   **Deep / Wide Research**:  For various search-oriented tasks.
-   **Webpage Generation**: Creating web pages from specific inputs.
-   **Trajectory Collection**:  Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides value for various user groups:

### For Agents Researchers & LLM Trainers
-   **Strong Baseline**: A powerful starting point, surpassing basic ReAct, for model training.
-   **One-Click Evaluation**: Simplified experimental processes through streamlined benchmarking.

### For Agent Application Developers
-   **Portable Scaffolding**: A proven framework for building real-world agent applications.
-   **Ease of Use**:  Simple scripts and a rich set of toolkits for quick starts.
-   **Modular Design**:  Highly customizable key components such as `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts
-   **Practical Examples**: Explore tasks like deep research reports, data analysis, and file organization.
-   **Simplicity & Debuggability**: The toolset and visual tracing makes development and debugging straightforward.

## 🧩 Core Concepts

*   **Agent**: An LLM with specific prompts, tools, and an environment.
*   **Toolkit**: A set of tools an agent utilizes.
*   **Environment**: The operating environment for the agent (e.g., browser, shell).
*   **ContextManager**: A module for managing the agent's context window.
*   **Benchmark**: A structured workflow for a specific dataset, including preprocessing, rollout, and evaluation logic.

Refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for further design and implementation details.

## 🚀 Getting Started

Youtu-Agent offers complete code and examples for a quick start. Run your first agent with these steps or use our streamlined Docker setup.

### Setup

#### Source Code Deployment

> \[!NOTE]
> Requires Python 3.12+. Use [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure the necessary API keys.
    ```

3.  Populate the `.env` file with API keys, like:

    ```bash
    # llm - OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    > [Tencent Cloud International](https://www.tencentcloud.com/) provides new DeepSeek API users with **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381). After applying, update the API key:

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker setup.

### Quick Start

Use the default agent, equipped with a search tool, to run a CLI chatbot:

```bash
# NOTE: Configure `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure `.env` with tool APIs:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Run the SVG image generation example:

```bash
python examples/svg_generator/main.py
```

For a web UI visualization, install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Run the web version:

```bash
python examples/svg_generator/main_web.py
```

Access the project using the local link displayed in the terminal:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will automatically search the web, collect information, and output an SVG visualization.

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

View results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Start quickly with a detailed guide.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Read our [**Contributing Guidelines**](./CONTRIBUTING.md) to improve Youtu-Agent.

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