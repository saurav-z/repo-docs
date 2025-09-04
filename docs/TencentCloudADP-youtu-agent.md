# Youtu-Agent: Build Powerful Autonomous Agents with Open-Source Models

**Youtu-Agent is your gateway to building high-performing, cost-effective autonomous agents, leveraging the power of open-source models.**  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent)

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#benchmark-performance"><b>🌟 Performance</b></a>
| <a href="#examples"><b>💡 Examples</b> </a>
| <a href="#features"><b>✨ Features</b> </a>
| <a href="#getting-started"><b>🚀 Getting Started</b> </a>
|
</p>

Youtu-Agent is a flexible and high-performance framework designed for building, running, and evaluating autonomous agents. This framework excels in key areas, e.g., data analysis, file processing, and in-depth research capabilities with open-source models.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features & Benefits:**

*   **Exceptional Performance:** Achieved impressive scores on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%), demonstrating strong capabilities with open-source DeepSeek-V3 models.
*   **Cost-Effective & Open-Source Focused:** Optimized for accessible and low-cost deployment, avoiding reliance on proprietary models.
*   **Practical Use Cases:** Offers out-of-the-box support for real-world tasks such as CSV analysis, literature review, file organization, and more.
*   **Flexible & Extensible Architecture:** Built upon [openai-agents](https://github.com/openai/openai-agents-python), supports diverse model APIs, tool integrations, and framework implementations.
*   **Simplified Automation:** Utilizes YAML-based configurations, automatic agent generation, and streamlined setup to reduce manual effort.

## 🗞️ News

*   🎁 **[2025-09-02]** Tencent Cloud International offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **[2025-08-28]** Live sharing updates about DeepSeek-V3.1 and its use in the `Youtu-Agent` framework. [Documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and lightweight tools, achieving impressive results on challenging benchmarks.

*   **WebWalkerQA:** Achieved 60.71% accuracy with `DeepSeek-V3-0324`, and 71.47% with the new `DeepSeek-V3.1`, setting a new SOTA.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`. Expanding evaluation to the full GAIA benchmark with multimodal tools.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images below to view detailed video demonstrations of Youtu-Agent in action:

<table>
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Data Analysis</strong><br>Analyze a CSV file and generate an HTML report.
        </td>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>File Management</strong><br>Rename and categorize local files.
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
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Wide Research</strong><br>Generate a comprehensive report by gathering extensive information, replicating Manus functionality.
        </td>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Paper Analysis</strong><br>Parse a given paper, perform analysis, and compile related literature.
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

### 🤖 Automatic Agent Generation

Youtu-Agent's standout feature is its ability to automatically generate agent configurations, eliminating the need for extensive coding or prompt engineering. Simply use YAML-based configs, allowing streamlined automation: A "meta-agent" interacts to capture requirements, then generates and saves the config.

```bash
# Clarify requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
    <tr>
        <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
            <strong>Automatic Agent Generation</strong><br>Interactively define your requirements, and automatically create and run agent configurations.
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

Explore more detailed examples and advanced use-cases in the [`examples`](./examples) directory, and find comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimalist Design:** The framework emphasizes simplicity and ease of use.
*   **Modular & Configurable:** Flexible customization and effortless integration of new components.
*   **Open-Source & Low-Cost:** Promotes accessibility and cost-effectiveness for a variety of applications.

### Core Features

*   **Built on openai-agents:** Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, benefiting from streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution, especially beneficial for benchmark evaluations.
*   **Tracing & Analysis System:** Beyond OTEL, the `DBTracingProcessor` provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation

*   **YAML-based Configuration:** Structured, easily manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generates agent configurations based on user requirements.
*   **Tool Generation & Optimization:** Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases

*   **Deep / Wide Research:** Covers common search-oriented tasks.
*   **Webpage Generation:** Examples include generating web pages based on specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research purposes.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides significant value for different user groups:

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

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For detailed design and implementation information, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent offers code and examples for a quick start. Run your first agent by following these steps, or use the streamlined Docker-based setup with an interactive frontend via [`docker/README.md`](./docker/README.md).

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+. Recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

Install Python and uv first.

Then clone and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # Fill in API keys in .env
```

Fill in the necessary keys, such as LLM API keys, in the `.env` file after copying `.env.example`.

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

> [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free.  Replace the API key in the .env file with the key after applying:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup with an interactive frontend.

### Quick Start

Use built-in configurations. The default config (`configs/agents/default.yaml`) defines an agent with a search tool:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Launch an interactive CLI chatbot:

```bash
# Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search.
# (Replace with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid search, run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Multiple ready-to-use examples are included. Configure tool APIs in the `.env` file under the tools module for examples requiring internet search:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Run this command to have the agent search the web and generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

To visualize the agent's runtime status using the web UI, install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```

Run the web version of the SVG image generation command:

```bash
python examples/svg_generator/main_web.py
```

Access the project once the terminal shows:

```bash
Server started at http://127.0.0.1:8848/
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent searches, collects, and outputs an SVG visualization given a research topic.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking. Evaluate on `WebWalkerQA`:

```bash
# Prepare dataset. This script will download and process WebWalkerQA dataset, and save it to DB.
python scripts/data/process_web_walker_qa.py

# Run evaluation with config `ww.yaml` with your custom `exp_id`. We choose the sampled small dataset `WebWalkerQA_15` for quick evaluation.
# NOTE: `JUDGE_LLM_TYPE, JUDGE_LLM_MODEL, JUDGE_LLM_BASE_URL, JUDGE_LLM_API_KEY` should be set in `.env`. Ref `.env.full`.
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

Results are stored and can be analyzed in the evaluation platform. See [Evaluation Analysis](./frontend/exp_analysis/README.md).

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project utilizes and builds upon the work of the following open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

If you find this project helpful, please cite it:

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