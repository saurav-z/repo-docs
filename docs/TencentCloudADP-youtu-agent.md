# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

> **Youtu-Agent empowers you to create advanced agents for data analysis, research, and more, all with the flexibility of open-source models.  [Explore the Youtu-Agent GitHub Repository](https://github.com/TencentCloudADP/youtu-agent)!**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a>
| <a href="README_JA.md"><b>日本語</b></a>
| <a href="#-benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#-examples"><b>💡 Examples</b> </a>
| <a href="#-features"><b>✨ Features</b> </a>
| <a href="#-getting-started"><b>🚀 Getting Started</b> </a>
| <a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

Youtu-Agent is a versatile and high-performance framework designed for constructing, executing, and evaluating autonomous agents. It offers powerful agent capabilities, including data analysis, file processing, and in-depth research, leveraging open-source models for accessible and cost-effective deployment.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **Superior Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% text-only subset, pass@1) using open-source DeepSeek-V3 series models.
*   ⚙️ **Open-Source Focused:** Optimized for deployment with open-source models, minimizing reliance on proprietary APIs and reducing costs.
*   🚀 **Practical Use Cases:**  Supports a range of applications, including CSV analysis, literature review, personal file organization, and upcoming podcast/video generation features.
*   💻 **Flexible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), offering extensibility for various model APIs (DeepSeek, GPT-OSS), tool integrations, and framework implementations.
*   ✨ **Simplified Workflow:** Utilizes YAML-based configurations, automatic agent generation, and a streamlined setup process to reduce manual effort.

## 📰 News

*   📺 [2025-09-09] Live sharing of design philosophy and basic usage: [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 [2025-09-02]  [Tencent Cloud International](https://www.tencentcloud.com/) offers DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Try it out](https://www.tencentcloud.com/document/product/1255/70381). For enterprise agent solutions, explore [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 [2025-08-28] Live sharing on DeepSeek-V3.1 and its use in Youtu-Agent: [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent excels on challenging benchmarks using open-source models and efficient tools.

*   **WebWalkerQA:**  Achieved 71.47% accuracy with DeepSeek-V3.1, setting a new state-of-the-art.  ([Dataset](https://huggingface.co/datasets/callanwu/WebWalkerQA))
*   **GAIA:**  Reached 72.8% pass@1 on the text-only validation subset using DeepSeek-V3-0324.  Ongoing work to expand evaluation to the full benchmark with multimodal tools.  ([Dataset](https://gaia-benchmark-leaderboard.hf.space/), [text-only subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json))

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images below for video demonstrations.

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
      <strong>Wide Research</strong><br>Generates a comprehensive report by gathering extensive information, replicating Manus.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a paper, performs analysis, and compiles related literature for results.
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
> Detailed examples are available in the [`examples`](./examples) directory and [documentation](https://tencentcloudadp.github.io/youtu-agent/examples/).

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with **automatic agent configuration generation.** Unlike other frameworks requiring coding or complex prompts, Youtu-Agent utilizes simple YAML-based configs for streamlined automation. A built-in "meta-agent" guides you through requirements gathering and generates the configuration automatically.

```bash
# Generate a config interactively
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Clarify requirements interactively, automatically generate an agent configuration, and run it immediately.
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
*   **Minimal Design:**  Focus on simplicity and ease of use, reducing unnecessary complexity.
*   **Modular & Configurable:**  Flexible customization and seamless integration of new components.
*   **Open-Source & Cost-Effective:**  Prioritizes accessibility and affordability for various applications.

### Core Features

*   **Built on openai-agents:** Leverages the [openai-agents](https://github.com/openai/openai-agents-python) SDK foundation, inheriting key capabilities such as streaming, tracing, and agent-loop, while ensuring compatibility with `responses` and `chat.completions` APIs. This setup allows for seamless adaptation to diverse models, including [gpt-oss](https://github.com/openai/gpt-oss).
*   **Asynchronous Execution:** Provides high-performance and efficient execution, especially when evaluating benchmarks.
*   **Comprehensive Tracing & Analysis:**  The `DBTracingProcessor` system offers detailed analysis of tool calls and agent trajectories, extending beyond OTEL (coming soon).

### Automation

*   **YAML-Based Configuration:**  Structured and easy-to-manage agent configurations.
*   **Automatic Agent Generation:**  Automatically generate agent configurations based on user input.
*   **Tool Generation & Optimization:**  Future support for tool evaluation, automated optimization, and custom tool generation.

### Use Cases

*   **In-Depth & Broad Research:**  Addresses common research-oriented tasks.
*   **Webpage Generation:**  Create web pages based on user-defined inputs.
*   **Trajectory Collection:**  Support data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is designed to benefit various user groups:

### For Agents Researchers & LLM Trainers

*   **Strong Baseline:**  A simple and robust baseline, surpassing basic ReAct, provides a strong starting point for model training and ablation studies.
*   **One-Click Evaluation:**  Streamlines experimentation and ensures consistent benchmarking with one-click evaluation scripts.

### For Agent Application Developers

*   **Proven Scaffolding:** A dependable and portable framework for creating real-world agent applications.
*   **Ease of Use:**  Quickly get started with simple scripts and a comprehensive collection of built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are well-encapsulated yet highly customizable.

### For AI & Agent Enthusiasts

*   **Practical Examples:**  The `/examples` directory showcases diverse tasks like in-depth research reports, data analysis, and personal file organization.
*   **Simplicity & Debugging:** Intuitive development and debugging facilitated by a rich toolkit and visual tracing tools.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** A set of tools that an agent can use.
*   **Environment:** The context in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for evaluating a dataset, including preprocessing, rollout, and evaluation logic.

For in-depth details on design and implementation, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent includes code and examples to help you get up and running quickly.

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+.  Recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

Ensure Python and uv are installed.

Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Configure the necessary API keys.
```

Fill in required API keys in the `.env` file, e.g., LLM API keys:

```bash
# llm - OpenAI API format compatibility required
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381). Replace the API key in the .env file after application:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with a web UI.

### Quick Start

Youtu-agent provides pre-configured agents.  The `configs/agents/simple/base_search.yaml` defines a simple agent using a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run an interactive CLI chatbot with:

```bash
# Configure SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (To be replaced with free alternatives)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 Learn more: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

The repository contains example use-cases.  You'll need to configure tool APIs in `.env` for examples using web search.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

For a web UI, download and install the frontend package from the Youtu-Agent releases.

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then, run the web version of the SVG generation command:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link shown in the terminal:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent automatically searches the web and generates an SVG visualization:

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmark evaluations. For example, evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. Download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and analyzed in the evaluation platform.  See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

Access the full documentation for in-depth information:

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: Get started quickly with a detailed guide.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions.

## 🙏 Acknowledgements

This project builds upon the work of these open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Contributions are welcome! Read our [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute.

## 📚 Citation

Cite this work as:

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