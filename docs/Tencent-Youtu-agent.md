# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

> Empower your projects with Youtu-Agent, a versatile and efficient framework for building and deploying AI agents, leveraging the power of open-source models like DeepSeek-V3.  ([View on GitHub](https://github.com/Tencent/Youtu-agent))

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<!-- <a href=https://arxiv.org/abs/2502.14345><img src=https://img.shields.io/badge/arXiv-2502.14345-b31b1b.svg></a> -->
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#-examples"><b>💡 Examples</b> </a>
| <a href="#-features"><b>✨ Features</b> </a>
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>


Youtu-Agent is a cutting-edge framework designed to streamline the development, execution, and evaluation of autonomous AI agents.  It empowers you to harness the capabilities of open-source models for a wide range of applications, including data analysis, file processing, and in-depth research.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Highlights:**

*   **High Performance:** Achieves state-of-the-art results, with 71.47% on WebWalkerQA and 72.8% on the GAIA text-only subset, showcasing the power of open-source DeepSeek-V3 models.
*   **Cost-Effective & Open-Source Focused:** Designed for accessible and affordable deployment, avoiding reliance on proprietary models.
*   **Practical Use Cases:** Offers out-of-the-box support for diverse tasks such as CSV analysis, literature reviews, personal file organization, and more.
*   **Flexible Architecture:** Built on the robust foundation of [openai-agents](https://github.com/openai/openai-agents-python), with extensibility for various model APIs (DeepSeek, gpt-oss) and tool integrations.
*   **Simplified Development:**  Leverages YAML-based configurations, automated agent generation, and streamlined setup for reduced manual effort.

## 📰 News

*   **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]** Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent is built on open-source models and lightweight tools, demonstrating strong results on challenging deep search and tool use benchmarks.

*   **WebWalkerQA**: Achieved 60.71% accuracy with `DeepSeek-V3-0324` and 71.47% with `DeepSeek-V3.1`.
*   **GAIA**: Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore the diverse capabilities of Youtu-Agent with these interactive examples:

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

Youtu-Agent simplifies agent creation with its automatic agent generation feature. Easily define your agent's requirements through simple YAML-based configs, eliminating the need for extensive code.

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

Explore more examples and advanced use cases in the [`examples`](./examples) directory and detailed documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Keeps the framework simple and user-friendly, minimizing overhead.
*   **Modular & Configurable:** Offers flexible customization and easy integration of new components.
*   **Open-Source & Low-Cost Focused:** Promotes accessibility and cost-effectiveness for diverse applications.

### Core Features

*   **Built on openai-agents:** Uses the openai-agents SDK for compatibility with `responses` and `chat.completions` APIs, supporting models like `gpt-oss`.
*   **Fully Asynchronous:** Enables high-performance execution, particularly beneficial for benchmark evaluations.
*   **Tracing & Analysis System:** Provides in-depth analysis of tool calls and agent trajectories via our `DBTracingProcessor` system. (Coming Soon)

### Automation

*   **YAML-Based Configuration:**  Provides structured and manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generates agent configurations based on user requirements.
*   **Tool Generation & Optimization:** Supports tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   **Deep/Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:**  Examples include web page generation based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent offers significant advantages for various users:

### For Agents Researchers & LLM Trainers

*   A **powerful baseline** for model training and ablation studies, surpassing basic ReAct methods.
*   **One-click evaluation scripts** for streamlined experimentation and consistent benchmarking.

### For Agent Application Developers

*   A **proven scaffolding** for building practical, real-world agent applications.
*   **Ease of Use**: Get started quickly with simple scripts and a comprehensive toolkit.
*   **Modular Design**: Offers high customization with key components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Use Cases**: Includes examples in the `/examples` directory for tasks such as deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability**: Utilizes a rich toolset and visual tracing for intuitive development and debugging.

## 🧩 Core Concepts

*   **Agent**: An LLM configured with prompts, tools, and an environment.
*   **Toolkit**: An encapsulated set of tools an agent can utilize.
*   **Environment**: The operational context of the agent (e.g., a browser, a shell).
*   **ContextManager**: A module for managing the agent's context window.
*   **Benchmark**: An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For in-depth information on design and implementation details, consult our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Get started with Youtu-Agent quickly with the following steps or refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+.  We recommend [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Install Python and uv.
2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure your API keys here.
```

Fill in the necessary API keys in the `.env` file, for example:

```bash
# llm
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

>   [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Once you’ve applied, replace the API key in the .env file below:

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

Use the built-in configuration to launch an interactive CLI chatbot with a search tool:

```bash
# NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid the search toolkit, run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` under the tools module:

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

or, run the web UI version:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```

```bash
python examples/svg_generator/main_web.py
```

Once deployed, access the project via the local link.

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

Results are stored and can be further analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project builds upon the excellent work of the following open-source projects:
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