# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent is a flexible and high-performance framework enabling the creation, execution, and evaluation of autonomous agents, all leveraging the power of open-source models.  [Explore the original repository](https://github.com/TencentCloudADP/Youtu-agent).**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

---

**Key Features:**

*   **Cutting-Edge Performance:** Achieves impressive results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1 on text-only subset) using open-source models like DeepSeek-V3.
*   **Open-Source & Cost-Effective:** Designed for accessible, low-cost deployment, eliminating the need for proprietary models.
*   **Versatile Use Cases:**  Includes out-of-the-box support for data analysis, file processing, literature review, and more, with podcast and video generation coming soon.
*   **Flexible Architecture:** Built upon the solid foundation of [openai-agents](https://github.com/openai/openai-agents-python), supporting various model APIs (DeepSeek, gpt-oss, and others), tool integrations, and framework implementations.
*   **Simplified Development:** YAML-based configuration, automated agent generation, and streamlined setup reduce manual effort.

---

## 📰 Latest News

*   **[2024-09-09]** Live sharing the design philosophy and basic usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   **[2024-09-02]**  Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381). For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2024-08-28]** Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🚀 Key Benefits & Highlights

*   **High Performance:**  Achieved strong results on challenging deep search and tool use benchmarks.
*   **Open-Source Models:**  Employs open-source models, promoting accessibility and cost-effectiveness.
*   **Ease of Use:** YAML-based configurations and automatic agent generation simplify agent creation.
*   **Practical Applications:**  Supports real-world tasks like data analysis, research, and file management.
*   **Extensible:** Compatible with various model APIs, offering flexibility in model selection.
*   **Community Support:**  Join the community and contribute to improving this framework.

## 🌟 Benchmark Performance

Youtu-Agent excels in benchmark performance, demonstrating its capabilities with open-source models.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA):**  Achieved 71.47% accuracy using DeepSeek-V3.1.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/):** Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using DeepSeek-V3-0324 (including models used within tools). We are actively extending evaluation to the full GAIA benchmark with multimodal tools.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore practical applications of Youtu-Agent through these examples:

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
> See the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/) for more details.

### 🤖 Automatic Agent Generation

A standout feature of `Youtu-Agent` is its ability to **automatically generate agent configurations**. In other frameworks, defining a task-specific agent often requires writing code or carefully crafting prompts. In contrast, `Youtu-Agent` uses simple YAML-based configs, which enables streamlined automation: a built-in "meta-agent" chats with you to capture requirements, then generates and saves the config automatically.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --config generated/xxx
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

> [!NOTE]
> See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy
- **Minimal design**: We try to keep the framework simple and easy to use, avoiding unnecessary overhead.
- **Modular & configurable**: Flexible customization and easy integration of new components.
- **Open-source model support & low-cost**: Promotes accessibility and cost-effectiveness for various applications.

### Core Features
- **Built on openai-agents**: Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
- **Fully asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
- **Tracing & analysis system**: Beyond OTEL, our `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
- **YAML based configuration**: Structured and easily manageable agent configurations.
- **Automatic agent generation**: Based on user requirements, agent configurations can be automatically generated.
- **Tool generation & optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases
- **Deep / Wide research**: Covers common search-oriented tasks.
- **Webpage generation**: Examples include generating web pages based on specific inputs.
- **Trajectory collection**: Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

`Youtu-Agent` is designed to provide significant value to different user groups:

### For Agents Researchers & LLM Trainers
- A **simple yet powerful baseline** that is stronger than basic ReAct, serving as an excellent starting point for model training and ablation studies.
- **One-click evaluation scripts** to streamline the experimental process and ensure consistent benchmarking.

### For Agent Application Developers
- A **proven and portable scaffolding** for building real-world agent applications.
- **Ease of Use**: Get started quickly with simple scripts and a rich set of built-in toolkits.
- **Modular Design**: Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts
- **Practical Use Cases**: The `/examples` directory includes tasks like deep research report generation, data analysis, and personal file organization.
- **Simplicity & Debuggability**: A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## 🧩 Core Concepts

- **Agent**: An LLM configured with specific prompts, tools, and an environment.
- **Toolkit**: An encapsulated set of tools that an agent can use.
- **Environment**: The world in which the agent operates (e.g., a browser, a shell).
- **ContextManager**: A configurable module for managing the agent's context window.
- **Benchmark**: An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more design and implementation details, please refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides complete code and examples to help you get started quickly. Follow the steps below to run your first agent, or refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

First, make sure Python and uv are installed.

Then clone the repository and sync dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

After copying the `.env.example` file, you need to fill in the necessary keys in the `.env` file, e.g. LLM API keys. For example:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Once you’ve applied, replace the API key in the .env file below:

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

Youtu-agent ships with built-in configurations. For example, the config `configs/agents/simple/base_search.yaml` defines a simple agent equipped with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

You can launch an interactive CLI chatbot with this agent by running:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

The repository provides multiple ready-to-use examples. Some examples require the agent to have internet search capabilities, so you’ll need to configure the tool APIs in the `.env` file under the tools module:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

For example, to enable the agent to automatically search online for information and generate an SVG image on the topic of “DeepSeek V3.1 New Features,” run the following command:

```bash
python examples/svg_generator/main.py
```

If you want to visualize the agent’s runtime status using the web UI, download the frontend package from the Youtu-Agent releases and install it locally:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Next, run the web version of the SVG image generation command:

```bash
python examples/svg_generator/main_web.py
```

Once the terminal shows the following message, the deployment is successful. You can access the project by clicking the local link:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

Given a research topic, the agent will automatically search the web, collect relevant information, and output an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent also supports benchmarking on standard datasets. For example, to evaluate on `WebWalkerQA`:

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

## 📖 Dive Deeper

After getting started, you can learn more about the framework and its capabilities through our full documentation:

- 📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
- 🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide to get you up and running.
- ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions and issues.

## 🙏 Acknowledgements

This project builds upon the excellent work of several open-source projects:
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

We welcome contributions from the community! If you'd like to help improve Youtu-Agent, please read our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

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