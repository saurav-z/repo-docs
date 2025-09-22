# 🤖 Youtu-Agent: Build, Run, and Evaluate Autonomous Agents with Open-Source Power

**Youtu-Agent empowers you to create powerful, efficient autonomous agents, revolutionizing tasks from data analysis to research, leveraging the power of open-source models.**  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/Youtu-agent)

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
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

Youtu-Agent is a versatile framework designed for building, running, and evaluating autonomous agents, particularly excelling with open-source LLMs.  It offers a flexible, high-performance environment for a variety of applications.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

## ✨ Key Features

*   ✅ **High Performance with Open-Source Models:** Achieve state-of-the-art results using open-source models, ensuring cost-effectiveness and accessibility.
    *   Achieved **71.47%** on WebWalkerQA (pass@1) and **72.8%** on GAIA (text-only subset, pass@1) with DeepSeek-V3 models, demonstrating strong capabilities.
*   ✅ **Cost-Effective & Accessible:** Designed for low-cost deployment without reliance on closed-source models, making it ideal for various projects.
*   ✅ **Practical Use Cases:** Built-in support for real-world applications such as CSV analysis, literature reviews, file organization, and content generation.
*   ✅ **Flexible Architecture:** Built upon the [openai-agents](https://github.com/openai/openai-agents-python), with adaptable support for diverse model APIs (DeepSeek, gpt-oss, etc.), various tool integrations, and different framework implementations.
*   ✅ **Streamlined Automation:** YAML-based configuration, automatic agent generation, and a simplified setup process reduce manual effort, allowing for rapid experimentation.

## 🗞️ News

*   📺 [2025-09-09] Live sharing: design philosophy and basic usage. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 [2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381)! For enterprise agent solutions, check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 [2025-08-28] Live sharing: DeepSeek-V3.1 updates and usage in `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent utilizes open-source models and lightweight tools to achieve impressive results on challenging benchmarks:

*   **WebWalkerQA:** Achieved **71.47%** accuracy with `DeepSeek-V3.1`.
*   **GAIA:** Scored **72.8%** pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.  (Evaluation on full GAIA benchmark with multimodal tools coming soon!)

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

(Click images for videos.)

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV and generates an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes files.
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
      <strong>Wide Research</strong><br>Generates a comprehensive report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses, analyzes, and compiles literature.
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
> More details available in the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automatic Agent Generation

A standout feature: `Youtu-Agent` can **automatically generate agent configurations** based on your requirements using simple YAML.  No more complex prompt engineering!

```bash
# Interactively define your needs and generate a config.
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Capture requirements, auto-generate configurations, and run immediately.
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
> Learn more at [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/).

## ✨ Features in Detail

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:**  Simple, easy-to-use framework minimizing overhead.
*   **Modular & Configurable:**  Flexible customization and easy integration of new components.
*   **Open-Source & Low-Cost:** Promoting accessibility and cost-effectiveness.

### Core Features

*   **Built on openai-agents:** Inherits robust capabilities from [openai-agents](https://github.com/openai/openai-agents-python), ensuring API compatibility and seamless model adaptation (e.g., gpt-oss).
*   **Fully Asynchronous:** Optimized for high-performance execution, especially crucial for benchmark evaluations.
*   **Tracing & Analysis System:** Provides detailed analysis of tool calls and agent trajectories (DBTracingProcessor - coming soon!).

### Automation

*   **YAML-Based Configuration:**  Structured, easily managed agent configurations.
*   **Automatic Agent Generation:**  Generate configurations based on user requirements.
*   **Tool Generation & Optimization:**  Future support for tool evaluation, optimization, and customization.

### Use Cases

*   Deep / Wide research
*   Webpage generation
*   Trajectory collection for training and research

## 🤔 Why Choose Youtu-Agent?

`Youtu-Agent` provides value for various user groups:

### For Researchers & LLM Trainers

*   A powerful baseline for model training and ablation studies.
*   One-click evaluation scripts to streamline the experimental process.

### For Agent Application Developers

*   A scaffolding for building real-world agent applications.
*   Ease of Use: Simple scripts and a rich set of built-in toolkits.
*   Modular Design: Highly customizable components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   Practical Use Cases: Examples in the `/examples` directory demonstrate agent capabilities.
*   Simplicity & Debuggability: Rich toolset and visual tracing tools for intuitive development.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools for agent use.
*   **Environment:** The world in which the agent operates.
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Workflow for a specific dataset.

For further implementation details, consult our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent offers complete code and examples for quick setup.  

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure the necessary API keys.
    ```

3.  Populate the `.env` file with your API keys (LLM, etc.).  For example:

    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    > [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381).

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker setup with a frontend.

### Quick Start

Use the `configs/agents/simple/base_search.yaml` config:

```bash
# NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in the `.env` file for examples requiring internet access:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

To generate an SVG on the topic of “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

To visualize results with a web UI:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link in the terminal, and you will be able to see the interactive frontend.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

To evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide for getting started.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

Thanks to these open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

See our [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute.

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