# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework empowering you to build, run, and evaluate autonomous agents, leveraging the power of open-source models. [Explore the Youtu-Agent GitHub Repository](https://github.com/Tencent/Youtu-agent)

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

<p align="center">
  <a href="README_ZH.md"><b>中文</b></a> |
  <a href="README_JA.md"><b>日本語</b></a> |
  <a href="#-benchmark-performance"><b>🌟 Performance</b></a> |
  <a href="#-examples"><b>💡 Examples</b></a> |
  <a href="#-features"><b>✨ Features</b></a> |
  <a href="#-getting-started"><b>🚀 Getting Started</b></a> |
  <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b></a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent offers a streamlined approach to creating intelligent agents capable of complex tasks like data analysis, file processing, and in-depth research, all while being cost-effective and open-source friendly.

**Key Features:**

*   ✅ **Exceptional Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8% on text-only subset) using open-source `DeepSeek-V3` models, demonstrating a strong open-source foundation.
*   💰 **Cost-Effective & Open-Source Focus:** Designed for accessible and low-cost deployment, minimizing reliance on proprietary models.
*   🛠️ **Practical Use Cases:** Includes out-of-the-box support for tasks like CSV analysis, literature reviews, file organization, and more, with podcast and video generation features coming soon.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting diverse model APIs (DeepSeek, GPT-OSS), tool integrations, and framework implementations.
*   🤖 **Automation & Simplicity:** YAML-based configurations, automatic agent generation, and streamlined setup reduce development time and effort.

## 🗞️ News

*   🎁 \[2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 \[2025-08-28] We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels on challenging deep search and tool use benchmarks.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**:  Achieved 71.47% accuracy with `DeepSeek-V3.1`, a new state-of-the-art result.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.  Multimodal tool evaluation is in progress.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click on the images to view detailed videos demonstrating Youtu-Agent's capabilities.

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

Youtu-Agent's standout feature is its **automatic agent generation**, simplifying agent configuration through YAML-based files.

```bash
# Interactively define your requirements and automatically generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively specify your requirements and have Youtu-Agent automatically create and run the agent configuration.
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

Explore the [`examples`](./examples) directory and the [`docs/examples.md`](./docs/examples.md) documentation for comprehensive examples and advanced use cases.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeps the framework simple and easy to use, avoiding unnecessary complexity.
*   **Modular & Configurable:** Offers flexible customization and easy integration of new components.
*   **Open-Source & Cost-Effective:** Promotes accessibility and cost-effectiveness for a wide range of applications.

### Core Features

*   **Built on openai-agents:** Leveraging [openai-agents](https://github.com/openai/openai-agents-python) SDK for streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs.
*   **Fully Asynchronous:** Enables high-performance and efficient execution, particularly beneficial for benchmark evaluations.
*   **Tracing & Analysis System:**  The `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories (coming soon).

### Automation

*   **YAML-Based Configuration:** Structured and easily manageable agent configurations.
*   **Automatic Agent Generation:** Based on user requirements, agent configurations can be automatically generated.
*   **Tool Generation & Optimization:**  Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide Research:** Handles common search-oriented tasks.
*   **Webpage Generation:** Examples include generating web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant benefits for different user groups:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** stronger than ReAct, ideal for model training and ablation studies.
*   **One-click evaluation scripts** for streamlined experimentation and consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable framework** for building real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and a rich toolkit.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Explore `/examples` for deep research, data analysis, and file organization applications.
*   **Simplicity & Debuggability:** A comprehensive toolset and visual tracing tools make development and debugging intuitive.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can utilize.
*   **Environment:** The context in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

Refer to the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for in-depth design and implementation details.

## 🚀 Getting Started

Youtu-Agent offers complete code and examples for a quick start.  Follow the steps below or use the Docker setup in [`docker/README.md`](./docker/README.md).

### Setup

#### Source Code Deployment

>   [!NOTE]
>   Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys in .env
```

Configure necessary API keys in the `.env` file. Example:

```bash
# llm config (DeepSeek)
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

>   [Tencent Cloud International](https://www.tencentcloud.com/) is offering free tokens (**Sep 1 – Oct 31, 2025**) for DeepSeek API users.  [Get Started](https://www.tencentcloud.com/document/product/1255/70381).  Then, update your `.env`:

```bash
# llm (Tencent Cloud DeepSeek)
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

See [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup.

### Quick Start

Youtu-Agent includes pre-built configurations, e.g., a default agent with a search tool (`configs/agents/default.yaml`):

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
# Configure SERPER_API_KEY and JINA_API_KEY in .env for web search.
python scripts/cli_chat.py --stream --config default
# Without the search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs (e.g., SERPER\_API\_KEY, JINA\_API\_KEY) in `.env` for examples using web search.

```bash
# tools
SERPER_API_KEY=<Get API Key>
JINA_API_KEY=<Get API Key>
```

To generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime in a web UI:

```bash
# Download the frontend package from the Youtu-Agent releases
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

# Run the web-enabled SVG generator:
python examples/svg_generator/main_web.py
```

Access the project at the local link after the "Server started" message.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will search the web and create an SVG visualization:

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking.  To evaluate on `WebWalkerQA`:

```bash
# Process the WebWalkerQA dataset
python scripts/data/process_web_walker_qa.py

# Run evaluation (using WebWalkerQA_15 for quick testing)
# Configure JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY in .env
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

For more information, explore the full documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)** Explore core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**  Get up and running quickly.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)** Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the work of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Contributions are welcome!  See our [**Contributing Guidelines**](./CONTRIBUTING.md).

## 📚 Citation

If you find this work useful, please cite:

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