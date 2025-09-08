# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework for building, running, and evaluating autonomous agents, achieving top benchmark performance with open-source models. Explore the [Youtu-Agent GitHub Repository](https://github.com/Tencent/Youtu-agent) for more information.

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#-key-features"><b>✨ Key Features</b> </a> 
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a> 
| <a href="#-examples"><b>💡 Examples</b> </a> 
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a> 
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a> 
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent empowers you to create powerful, efficient, and cost-effective autonomous agents using open-source models.

## ✨ Key Features

*   **Superior Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% on text-only subset, pass@1) using open-source DeepSeek-V3 series models.
*   **Open-Source & Cost-Effective:** Designed for accessibility, minimizing reliance on closed models and reducing deployment costs.
*   **Versatile Use Cases:**  Out-of-the-box support for data analysis, file processing, literature review, and more.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), with support for diverse model APIs, tool integrations, and framework implementations.
*   **Simplified Development:**  Leverages YAML-based configurations, automatic agent generation, and streamlined setup for reduced manual effort.

## 🗞️ News

*   **Free DeepSeek Tokens:**  Tencent Cloud International is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **DeepSeek-V3.1 Updates:**  Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [Documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels in benchmark performance using open-source models and lightweight tools.

*   **WebWalkerQA:** Achieved 71.47% accuracy (pass@1) with DeepSeek-V3.1, setting a new SOTA.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324. (More info on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json))

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click on the images to view detailed videos showcasing Youtu-Agent's capabilities.

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
      <video src="https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775" 
             poster="https://img.youtube.com/vi/SCR4Ru8_h5Q/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files.
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
      <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" 
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height=300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses and analyzes a given paper, compiling related literature.
      <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" 
             poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its automatic agent generation feature. Define tasks using simple YAML configurations, and the "meta-agent" will generate the configuration for you.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Clarify requirements, automatically generate agent config, and run immediately.
    </td>
    <td style="border: 1px solid black; padding:10px; vertical-align:top; width: 400px;">
      <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" 
             poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="auto" 
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>


Explore further examples and advanced use cases in the [`examples`](./examples) directory and the comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant benefits for:

*   **Agents Researchers & LLM Trainers:** A simple yet powerful baseline for model training and ablation studies.
*   **Agent Application Developers:** A proven and portable framework for building real-world agent applications with ease of use and a modular design.
*   **AI & Agent Enthusiasts:** Practical use cases, simplicity, and debuggability via a rich toolset and visual tracing tools.

## 🧩 Core Concepts

*   **Agent:** An LLM with prompts, tools, and an environment.
*   **Toolkit:**  An encapsulated set of tools for agents.
*   **Environment:** Where the agent operates (browser, shell, etc.).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more detailed information, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to get started with Youtu-Agent:

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and uv for dependency management.

1.  Install Python and uv.
2.  Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure your API keys.
    ```

3.  Configure API keys in the `.env` file, including LLM API keys.

    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    **Important:**  Tencent Cloud International is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381).

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup.

### Quick Start

Run a basic agent:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore Examples

Configure tool APIs in `.env` (e.g., `SERPER_API_KEY`, `JINA_API_KEY`) to run examples using web search.

Example: Generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

Use the web UI:

1.  Download and install the frontend package:

    ```bash
    curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl
    uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
    ```

2.  Run the web version of the SVG generation example:

    ```bash
    python examples/svg_generator/main_web.py
    ```

    Access the project via the local link displayed in the terminal.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on datasets like `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored for analysis.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project is built upon the work of:

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