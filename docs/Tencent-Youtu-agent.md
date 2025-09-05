# 🤖 Youtu-Agent: Build and Deploy Powerful AI Agents with Open-Source Models

Youtu-Agent is a flexible and high-performance framework for creating autonomous agents, offering advanced capabilities like data analysis and file processing, all powered by open-source models.  [Explore the Youtu-Agent Repository](https://github.com/Tencent/Youtu-agent)

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#key-features"><b>✨ Key Features</b> </a> 
| <a href="#benchmark-performance"><b>🌟 Performance</b></a> 
| <a href="#examples"><b>💡 Examples</b> </a> 
| <a href="#getting-started"><b>🚀 Getting Started</b> </a> 
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a> 
</p>

Youtu-Agent empowers developers and researchers to build cutting-edge AI agents using open-source models, achieving strong performance on benchmarks like WebWalkerQA and GAIA.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

## ✨ Key Features

*   **High Performance**:  Achieves impressive results on benchmarks, including 71.47% on WebWalkerQA and 72.8% on GAIA (text-only subset), demonstrating the power of open-source models.
*   **Open-Source & Cost-Effective**: Designed for accessibility and low-cost deployment, avoiding reliance on proprietary models.
*   **Practical Use Cases**: Supports a variety of tasks, including CSV analysis, literature review, file organization, and upcoming podcast/video generation.
*   **Flexible Architecture**: Built on [openai-agents](https://github.com/openai/openai-agents-python), enabling diverse model API support (DeepSeek, gpt-oss), and easy integration of tools and frameworks.
*   **Automated Agent Generation**:  YAML-based configurations and a built-in meta-agent simplify agent creation and deployment.

## 🗞️ News

*   **[2025-09-02]**: [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]**: Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework, along with shared [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent showcases strong performance on deep search and tool use benchmarks using open-source models and lightweight tools.

*   **WebWalkerQA**: Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art (SOTA) performance.
*   **GAIA**: Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.  Multimodal tools and full GAIA benchmark evaluation is in progress.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the capabilities of Youtu-Agent through these interactive examples.  Click the images to view detailed videos.

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

A standout feature of `Youtu-Agent` is its ability to **automatically generate agent configurations**. In other frameworks, defining a task-specific agent often requires writing code or carefully crafting prompts. In contrast, `Youtu-Agent` uses simple YAML-based configs, which enables streamlined automation: a built-in "meta-agent" chats with you to capture requirements, then generates and saves the config automatically.

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

For more detailed examples and advanced use-cases, please refer to the [`examples`](./examples) directory and our comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## 🧩 Core Concepts

*   **Agent**: An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit**: An encapsulated set of tools that an agent can use.
*   **Environment**: The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager**: A configurable module for managing the agent's context window.
*   **Benchmark**: An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is a versatile framework suitable for various users:

### For Agents Researchers & LLM Trainers

*   A **simple yet powerful baseline** that is stronger than basic ReAct, serving as an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Use Cases**: The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability**: A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

For more design and implementation details, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to quickly deploy your first AI agent.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Install Prerequisites:** Ensure Python and `uv` are installed.
2.  **Clone Repository and Sync Dependencies:**

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # NOTE: You should then config the necessary API keys.
    ```
3.  **Configure API Keys:**  Populate the `.env` file with the necessary API keys for your chosen LLM (e.g., DeepSeek) and tools. Replace the placeholder keys in `.env` with your actual credentials.

    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    > [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Once you’ve applied, replace the API key in the .env file below:

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Please refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with interactive frontend.

### Quick Start

Use the pre-configured agent:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

1.  **Configure Tool APIs:** In `.env`, set up API keys for the tools used in the examples, such as `SERPER_API_KEY` and `JINA_API_KEY`.
2.  **Run Example:** For instance, generate an SVG image based on a web search with the command:

    ```bash
    python examples/svg_generator/main.py
    ```
3.  **Run the Web UI version**
    1.  Download the frontend package from the Youtu-Agent releases.
    2.  Install the frontend package:

        ```bash
        uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
        ```
    3.  Run the web version of the SVG image generation command:

        ```bash
        python examples/svg_generator/main_web.py
        ```
        Access the project by clicking the local link that will be presented.

    ![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

    ![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Test your agent on standard datasets. For example:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and can be further analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project is built upon the foundation of these exceptional open-source projects:
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

If you find this work useful, please consider citing:

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