# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent is a flexible and high-performance framework that empowers you to build, run, and evaluate autonomous agents, all leveraging the power of open-source models.**  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent)

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

Youtu-Agent provides a robust foundation for agent development, offering impressive performance and a cost-effective approach. Key features include:

*   **🏆 High Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% on text-only subset, pass@1) using only open-source models.
*   **🔓 Open-Source & Cost-Aware:** Designed for accessibility, enabling deployment with free or low-cost models.
*   **🛠️ Practical Use Cases:**  Supports diverse tasks out-of-the-box, including CSV analysis, literature review, and file organization (more coming soon).
*   **⚙️ Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), and extensible to a wide range of models and tool integrations.
*   **🚀 Automation:** Simplifies agent creation and management with YAML-based configurations and automatic agent generation.

## 🗞️ News

*   **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]**  We held a live sharing session on DeepSeek-V3.1 and its integration with the `Youtu-Agent` framework.  [Documentation from the session](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels on deep search and tool use benchmarks:

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA):**  71.47% accuracy with DeepSeek-V3.1, demonstrating cutting-edge performance.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/):** Achieved 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.  Multimodal tool support is coming soon!

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

See the power of Youtu-Agent in action:

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files.
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
      <strong>Wide Research</strong><br>Generates a comprehensive report, similar to Manus.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyzes a paper, performs related literature research, and compiles results.
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

Youtu-Agent simplifies agent creation with automated configuration:

```bash
# Interactively define requirements and generate config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br> Define your requirements and automatically generate and run agent configurations.
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

Explore more examples in the [`examples`](./examples) directory and comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeps the framework streamlined and easy to use.
*   **Modular and Configurable:** Facilitates customization and easy integration of new components.
*   **Open-Source & Cost-Effective:** Promotes accessibility and affordability.

### Core Features

*   **Built on openai-agents:** Leverages the openai-agents SDK for robust features.
*   **Fully Asynchronous:** Enables high-performance, especially beneficial for benchmark evaluation.
*   **Tracing & Analysis:** The `DBTracingProcessor` provides deep insights into tool calls and agent trajectories (coming soon).

### Automation

*   **YAML Configuration:** Simplifies agent management with structured YAML files.
*   **Automatic Agent Generation:** Automates configuration generation based on user input.
*   **Tool Generation & Optimization:** Plans for future support of tool evaluation and optimization, including customized tool creation.

### Use Cases

*   **Deep / Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:** Examples include generating web pages.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed for:

### For Agents Researchers & LLM Trainers

*   A **strong baseline** for model training and ablation studies, exceeding basic ReAct.
*   **One-click evaluation scripts** for simplified experimentation.

### For Agent Application Developers

*   A **reliable scaffolding** for building real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and built-in toolkits.
*   **Modular Design:**  Easily customize and extend key components.

### For AI & Agent Enthusiasts

*   **Practical Use Cases:**  Explore tasks like deep research and data analysis in the `/examples` directory.
*   **Simplicity & Debuggability:** The rich toolset and visual tracing make development intuitive.

## 🧩 Core Concepts

*   **Agent:** An LLM with specific prompts, tools, and an environment.
*   **Toolkit:** A collection of tools accessible to an agent.
*   **Environment:** The context in which an agent operates (e.g., browser).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** Encapsulates a workflow for specific datasets, including preprocessing and evaluation.

For more information, see our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent is easy to get up and running. Follow these steps or use the Docker setup in [`docker/README.md`](./docker/README.md).

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys here
```

*   Configure the required API keys in the `.env` file (e.g., LLM API keys).

    ```bash
    # LLM config (replace with your values)
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    >   [Tencent Cloud International](https://www.tencentcloud.com/) offers new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  Use it to avoid API costs.  If you use this method replace the key in the .env with the one obtained from the Tencent Cloud console:

    ```bash
    # LLM
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup.

### Quick Start

Use the default agent configuration:

```bash
# Requires SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (We will provide free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# Run the base agent if you want to avoid using search:
python scripts/cli_chat.py --stream --config base
```

📖 [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

1.  Configure tool APIs in the `.env` file for examples requiring internet search.

    ```bash
    # tools
    SERPER_API_KEY=<your_serper_api_key>
    JINA_API_KEY=<your_jina_api_key>
    ```

2.  Run the SVG image generation example:

    ```bash
    python examples/svg_generator/main.py
    ```

3.  (Optional) Run the web UI example:

    ```bash
    # Download the frontend package.
    curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

    # Install the frontend package
    uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
    python examples/svg_generator/main_web.py
    ```

    Open the provided local link.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)
![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

```bash
# Prepare dataset
python scripts/data/process_web_walker_qa.py

# Run eval (replace with your values)
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

View results in the evaluation platform.  See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)
![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project is built on the foundations of these excellent open-source projects:

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