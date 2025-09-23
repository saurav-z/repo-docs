# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Empower your projects with Youtu-Agent, a flexible and high-performing framework for creating, running, and evaluating AI agents, built to excel with open-source models.  ([View the original repository](https://github.com/Tencent/Youtu-agent))**

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

Youtu-Agent provides a robust and accessible foundation for creating AI agents, supporting diverse applications such as data analysis, file processing, and research, all while prioritizing open-source models and cost-effectiveness.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **Superior Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%, text-only subset) using open-source DeepSeek-V3 models.
*   💸 **Cost-Effective & Open-Source Friendly:** Designed for accessible, low-cost deployment without dependence on proprietary models.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting a variety of models (DeepSeek, gpt-oss) and tool integrations.
*   🚀 **Automated Agent Generation:** YAML-based configuration and auto-generation streamline agent creation and deployment, reducing manual setup.
*   💡 **Practical Use Cases:** Out-of-the-box support for tasks like CSV analysis, literature review, file organization, and multimedia generation (coming soon).

## 📰 News

*   **\[2025-09-09]** Live sharing of `Youtu-Agent`'s design and usage. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)]
*   **\[2025-09-02]** Free DeepSeek API tokens (**3 million free tokens**) from [Tencent Cloud International](https://www.tencentcloud.com/) for new users, until **Oct 31, 2025**. [Try it out](https://www.tencentcloud.com/document/product/1255/70381). For enterprise agent solutions, see [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **\[2025-08-28]** Live sharing updates about DeepSeek-V3.1 and `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)]

## 🌟 Benchmark Performance

Youtu-Agent demonstrates strong results with open-source models on deep search and tool use benchmarks.

*   **WebWalkerQA:** Achieved **71.47%** accuracy using `DeepSeek-V3.1`, setting a new SOTA. (See [WebWalkerQA dataset](https://huggingface.co/datasets/callanwu/WebWalkerQA))
*   **GAIA:** Achieved **72.8%** pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`. Evaluation on the full GAIA benchmark with multimodal tools is ongoing.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images to view detailed video demonstrations.

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyze CSV data and generate HTML reports.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Rename and categorize local files automatically.
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
      <strong>Wide Research</strong><br>Generate comprehensive reports by gathering extensive information.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyze papers and compile related literature for summaries.
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

>   **[NOTE]** See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

Youtu-Agent's automated agent generation streamlines the creation process.  Simply provide your requirements, and the system will generate YAML configurations automatically.

```bash
# Clarify requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interact with the system to generate and run agent configurations.
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

>   **[NOTE]** See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Focuses on simplicity and ease of use.
*   **Modular & Configurable:** Allows easy customization and integration.
*   **Open-Source Model Support & Low-Cost:** Promotes accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** High-performance execution, crucial for benchmarking.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories. (coming soon)

### Automation

*   **YAML-Based Configuration:** Structured configurations for easy management.
*   **Automatic Agent Generation:** Generate agent configurations based on user input.
*   **Tool Generation & Optimization:** Support for tool evaluation, automated optimization, and customized tool generation is planned.

### Use Cases

*   **Deep / Wide Research:** Ideal for search-oriented tasks.
*   **Webpage Generation:** Generate web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant value to different user groups:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline**, stronger than basic ReAct, for model training and research.
*   **One-click evaluation scripts** to streamline benchmarking.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:** Quick setup with simple scripts and rich toolkits.
*   **Modular Design:** Highly customizable components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:** Deep research, data analysis, and file organization examples.
*   **Simplicity & Debuggability:** Rich toolset and visual tracing tools for development and debugging.

## 🧩 Core Concepts

*   **Agent:** An LLM with specific prompts, tools, and an environment.
*   **Toolkit:** A set of tools an agent can use.
*   **Environment:** Where the agent operates (e.g., a browser, a shell).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** An encapsulated workflow for a dataset, including processing and judging logic.

Refer to the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/) for details.

## 🚀 Getting Started

Follow these steps to get started:

### Setup

#### Source Code Deployment

>   **[NOTE]** Requires Python 3.12+ and we recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and `uv`.
2.  Clone the repository:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure API keys here.
    ```

3.  Configure API keys in `.env` (e.g., LLM API keys):

    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    >   [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381). Once applied, replace the API key in the .env:

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

See [`docker/README.md`](./docker/README.md) for Docker setup.

### Quick Start

Run the `simple/base_search` agent:

```bash
# Configure SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (We plan to replace these with free alternatives)
python scripts/cli_chat.py --config simple/base_search
# To avoid the search toolkit:
python scripts/cli_chat.py --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Enable the agent to search the web and generate an SVG image:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime in the web UI:

```bash
# Download frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl

# Run the web version
python examples/svg_generator/main_web.py
```

Access the project via the local link provided in the terminal.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on `WebWalkerQA`:

```bash
# Prepare dataset (download and process WebWalkerQA)
python scripts/data/process_web_walker_qa.py

# Run evaluation
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze the results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts and features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Detailed guide for getting started.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Answers to common questions.

## 🙏 Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

See our [**Contributing Guidelines**](./CONTRIBUTING.md) for details on how to contribute.

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