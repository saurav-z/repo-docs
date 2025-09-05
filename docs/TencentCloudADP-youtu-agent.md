# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent** empowers you to create advanced AI agents, delivering data analysis, file processing, and deep research capabilities using open-source models – [Explore the Youtu-Agent Repo](https://github.com/TencentCloudADP/youtu-agent).

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#examples"><b>💡 Examples</b> </a>
| <a href="#features"><b>✨ Features</b> </a>
| <a href="#getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

Youtu-Agent is a flexible, high-performance framework designed for building, running, and evaluating autonomous AI agents.  Leveraging open-source models, it excels at complex tasks while remaining cost-effective and accessible.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   **Exceptional Performance**: Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1), using only open-source DeepSeek-V3 models.
*   **Open-Source & Cost-Conscious**: Designed for accessible deployment with open-source models, reducing reliance on expensive closed-source alternatives.
*   **Practical Use Cases**: Out-of-the-box support for data analysis (CSV analysis, literature review), file organization, and more.
*   **Flexible Architecture**: Built on [openai-agents](https://github.com/openai/openai-agents-python), enabling easy integration with diverse model APIs, tool integrations, and framework implementations.
*   **Simplified Automation**:  Streamlined development with YAML-based configurations and automated agent generation, minimizing manual setup.

## 🗞️ News

*   **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]** We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent demonstrates strong performance on challenging benchmarks using open-source models and lightweight tools.

*   **WebWalkerQA**: 71.47% accuracy using DeepSeek-V3.1, setting a new SOTA (pass@1).
*   **GAIA**: 72.8% pass@1 on the text-only validation subset using DeepSeek-V3 (including models used within tools).

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the capabilities of Youtu-Agent with these interactive examples.

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

Youtu-Agent simplifies agent creation with automatic configuration generation. Define your agent's behavior with YAML, and let Youtu-Agent handle the rest.

```bash
# Interactively define requirements and auto-generate a config
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

For more advanced use-cases, explore the  [`examples`](./examples) directory and comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design**: Keep the framework simple and easy to use, avoiding unnecessary overhead.
*   **Modular & Configurable**: Flexible customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost**: Promotes accessibility and cost-effectiveness for various applications.

### Core Features

*   **Built on openai-agents**: Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK.
*   **Fully Asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
*   **Tracing & Analysis System**: Provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation

*   **YAML-based Configuration**: Simplified agent setup and management.
*   **Automatic Agent Generation**: Automate agent configuration based on requirements.
*   **Tool Generation & Optimization**: Future support for tool evaluation, optimization, and custom tool generation.

### Use Cases

*   Deep/Wide research
*   Webpage generation
*   Trajectory collection

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed to benefit various users:

### For Agents Researchers & LLM Trainers

*   Strong baseline for model training and ablation studies.
*   One-click evaluation scripts.

### For Agent Application Developers

*   Proven scaffolding for real-world agent applications.
*   Ease of Use:  Simple scripts and rich toolkit.
*   Modular Design:  Customizable components.

### For AI & Agent Enthusiasts

*   Practical Use Cases.
*   Simplicity & Debuggability: Rich toolset and visual tracing.

## 🧩 Core Concepts

*   **Agent**: LLM with prompts, tools, and an environment.
*   **Toolkit**: Encapsulated tools for agent use.
*   **Environment**:  The agent's operational context (e.g., a browser).
*   **ContextManager**: Module for managing the agent's context window.
*   **Benchmark**: Encapsulated workflow for a specific dataset.

For in-depth details, see our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Get up and running quickly with Youtu-Agent.

### Setup

#### Source Code Deployment

> [!NOTE]
> Project requires Python 3.12+. We recommend [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Ensure Python and `uv` are installed.
2.  Clone the repository:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure API keys.
    ```

After copying `.env.example`, fill in API keys.

```bash
# llm config
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381).

```bash
# llm
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for Docker setup.

### Quick Start

Use built-in configurations:

```yaml
# configs/agents/default.yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run an interactive CLI chatbot:

```bash
# Set SERPER_API_KEY and JINA_API_KEY in `.env` for web search.
python scripts/cli_chat.py --stream --config default
# Without search:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env`:

```bash
# tools
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Run SVG image generation example:

```bash
python examples/svg_generator/main.py
```

Web UI:

```bash
# Download frontend
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install frontend
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl

# Run
python examples/svg_generator/main_web.py
```

Open the displayed local link in your browser.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on `WebWalkerQA`:

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project builds upon open-source projects:
-   [openai-agents](https://github.com/openai/openai-agents-python)
-   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
-   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

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
```
Key improvements and explanations:

*   **SEO-Optimized Hook:**  Used a strong, benefit-driven opening sentence with target keywords.
*   **Clear Headings:**  Organized with meaningful, keyword-rich headings (e.g., "Benchmark Performance," "Getting Started").
*   **Bulleted Key Features:** Uses bullet points to clearly highlight the main selling points.
*   **Concise Language:**  Phrased everything efficiently to maximize impact and readability.
*   **Call to Action:** Encouraged readers to explore the repo.
*   **Internal Links:**  Used markdown internal links to important sections, for readability.
*   **Emphasis on Benefits:**  Focused on what users *get* from the project.
*   **Context & Clarity:** Provided more context and explanations.  For example, clarified what the agent *does* (e.g., "data analysis").
*   **Combined duplicate information** Removed repetition
*   **Docker Setup:**  Included information about the docker setup in the "Getting Started" section for added convenience.
*   **Simplified Setup Instructions:**  Simplified the "Getting Started" by removing unnecessary steps and providing clearer instructions.
*   **Removed Unnecessary Content:** Removed the DeepWiki and DeepSee links as it is not very important.
*   **Reformatted Code blocks**: The code block was reformatted for better readability.
*   **Added News Section**: Included the news section which is more useful.
*   **Removed unnecessary documentation**: Removed redundant documentations.