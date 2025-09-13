# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

Youtu-Agent is a versatile agent framework empowering developers to create and deploy cutting-edge AI agents, backed by high-performance open-source models.  Explore the capabilities and get started today!  [See the original repository](https://github.com/Tencent/Youtu-agent).

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
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

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

## Key Features

*   **High-Performance Open-Source Agents:** Achieve impressive results on benchmarks like WebWalkerQA and GAIA using models like DeepSeek-V3, optimizing for cost-effective deployment.
*   **Practical Applications:** Out-of-the-box support for diverse tasks including data analysis, file processing, and research, with more on the way.
*   **Flexible and Extensible:** Built on the foundation of [openai-agents](https://github.com/openai/openai-agents-python), Youtu-Agent supports a wide array of models, tools, and frameworks.
*   **Automated Configuration:** YAML-based configuration, automatic agent generation and streamlined setup, dramatically reduces manual effort.

## 🗞️ News

*   🎁 **[Sep 2, 2025]**  Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381).  For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[Aug 28, 2025]**  Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent excels on challenging deep search and tool use benchmarks with open-source models.

*   **WebWalkerQA**: Achieved **71.47%** accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art.
*   **GAIA**: Achieved **72.8%** pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324` (including models used within tools).

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images below to see detailed demonstrations:

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
      <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a given paper and compiles related literature.
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

Quickly generate agent configurations with YAML-based configs.  A built-in meta-agent guides you through requirements, automatically producing and saving the config.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define requirements, auto-generate agent configs, and run them instantly.
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
*   **Minimal Design:** Simple and easy to use, avoiding unnecessary complexity.
*   **Modular & Configurable:** Highly flexible customization and seamless component integration.
*   **Open-Source & Cost-Effective:** Promotes accessibility and low deployment costs.

### Core Features

*   **Built on openai-agents:** Leverages the openai-agents SDK, providing streaming, tracing, and agent-loop capabilities, and API compatibility.
*   **Fully Asynchronous:** High-performance execution, especially for benchmarking.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories. (Coming soon)

### Automation

*   **YAML-based Configuration:**  Structured and easily managed agent configurations.
*   **Automatic Agent Generation:** Creates agent configurations based on user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, automated optimization, and custom tool generation.

### Use Cases

*   **Deep / Wide Research:** Handles common search-oriented tasks.
*   **Webpage Generation:** Generate webpages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed for:

### Agent Researchers & LLM Trainers

*   A **simple yet powerful baseline** surpassing ReAct, providing an excellent starting point for model training and ablation studies.
*   **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### Agent Application Developers

*   A **proven and portable scaffolding** for building real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### AI & Agent Enthusiasts

*   **Practical Use Cases:** The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools for agents.
*   **Environment:** The agent's operating environment (e.g., a browser, a shell).
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for dataset evaluation, including preprocessing, rollout, and judging logic.

Detailed design and implementation are available in our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to run your first agent, or use the Docker setup for a streamlined experience.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

Ensure Python and `uv` are installed.

Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure API keys here.
```

Configure the `.env` file with necessary API keys. For example:

```bash
# llm - OpenAI compatible
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

>  Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace the API key in the `.env` file:

```bash
# llm
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for Docker-based setup.

### Quick Start

Youtu-Agent provides built-in configurations. The default config (`configs/agents/default.yaml`) defines a search agent:

```yaml
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
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search access.
python scripts/cli_chat.py --stream --config default
# To exclude the search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env`:

```bash
# tools
SERPER_API_KEY=<get API Key from https://serper.dev/playground>
JINA_API_KEY=<get API Key from https://jina.ai/reader>
```

Example: Generate an SVG image on the topic of “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime status with a web UI, download and install the frontend:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Run the web version of the SVG image generation example:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link in the terminal.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent automatically searches the web and outputs an SVG visualization:

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking. Example: Evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. Downloads and processes WebWalkerQA, saving it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`.
# Choose `WebWalkerQA_15` for quick evaluation.
# Ensure `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` are in `.env`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Analyze results in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Explore the full documentation:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

## 🙏 Acknowledgements

This project is built on:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Contributions are welcome! Read the [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

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