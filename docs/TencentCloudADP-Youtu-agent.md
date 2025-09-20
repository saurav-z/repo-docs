<!-- Improved README.md for Youtu-Agent -->

# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Unlock the potential of AI agents with Youtu-Agent, a high-performance, flexible framework that empowers you to create, run, and evaluate intelligent agents using open-source models.** ([View on GitHub](https://github.com/TencentCloudADP/Youtu-agent))

<div align="center">
<a href="https://tencentcloudadp.github.io/youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文</b></a> | <a href="README_JA.md"><b>日本語</b></a> |
<a href="#key-features"><b>✨ Features</b></a> | <a href="#getting-started"><b>🚀 Getting Started</b></a> | <a href="#examples"><b>💡 Examples</b> </a> | <a href="#-benchmark-performance"><b>🌟 Performance</b></a> |
<a href="https://discord.gg/svwuqgUx"><b>📢 Join Community</b> </a>
</p>

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent revolutionizes agent development, offering a streamlined, cost-effective, and powerful solution for building autonomous agents using open-source models, delivering exceptional results in tasks like data analysis, research, and more.

## Key Features

*   ✅ **State-of-the-Art Performance:** Achieves impressive results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% on text-only, pass@1) using open-source DeepSeek-V3 models, demonstrating its competitive edge.
*   💰 **Cost-Effective & Open Source:** Designed for accessible and affordable deployment, avoiding reliance on closed models.
*   🛠️ **Practical Use Cases:** Offers built-in support for diverse applications, including CSV analysis, literature review, file organization, and soon, podcast/video generation.
*   ⚙️ **Flexible Architecture:** Built on the robust [openai-agents](https://github.com/openai/openai-agents-python) foundation, offering seamless integration with various model APIs (DeepSeek to gpt-oss), tool integrations, and framework implementations.
*   🤖 **Automation & Simplicity:** Streamlines agent creation and management with YAML-based configurations, automatic agent generation, and simplified setup, significantly reducing manual effort.

## 🗞️ News

*   📺 **September 9, 2025:** Live sharing the design philosophy and basic usage of `Youtu-Agent`. [[video](https://www.bilibili.com/video/BV1mypqz4EvS)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNLgt3CbnxRWaYWnW4)].
*   🎁 **September 2, 2025:** [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 **August 28, 2025:** Live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. [[video](https://www.bilibili.com/video/BV1XwayzrETi/)] [[documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF)].

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and efficient tools, delivering strong results on challenging benchmarks:

*   **WebWalkerQA**: Achieved 71.47% accuracy with `DeepSeek-V3.1`.
*   **GAIA**: Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore Youtu-Agent's capabilities through these example applications.  Click the images for video demonstrations.

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes a CSV file and generates an HTML report.
      <video src="https://github.com/user-attachments/assets/60193435-b89d-47d3-8153-5799d6ff2920" 
             poster="https://img.youtube.com/vi/r9we4m1cB6M/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Renames and categorizes local files for the user.
       <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" 
             poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Wide Research</strong><br>Gathers extensive information to generate a comprehensive report, replicating the functionality of Manus.
        <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" 
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses a given paper, performs analysis, and compiles related literature to produce a final result.
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

Youtu-Agent streamlines agent creation with its automatic agent generation feature.  Simply provide your requirements and let Youtu-Agent create the agent config for you.

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

> [!NOTE]
> See [documentation](https://tencentcloudadp.github.io/youtu-agent/auto_generation/) for more details.

## ✨ Features (Detailed)

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:** Simple and easy-to-use framework.
*   **Modular & Configurable:** Easy customization and component integration.
*   **Open-Source & Low-Cost:** Promotes accessibility and affordability.

### Core Features

*   **Based on openai-agents:** Utilizing the features of [openai-agents](https://github.com/openai/openai-agents-python), including streaming, tracing, and agent-loop, and is compatible with the `responses` and `chat.completions` APIs for adapting to models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enabling high performance, particularly for benchmark evaluations.
*   **Tracing & Analysis System:** In-depth analysis of tool calls and agent trajectories via the `DBTracingProcessor`. (coming soon)

### Automation

*   **YAML-Based Configuration:** Organized and manageable agent configurations.
*   **Automatic Agent Generation:** Automatically generates agent configurations from user requirements.
*   **Tool Generation & Optimization:** Future support for tool evaluation, optimization, and customized tool creation.

### Use Cases

*   **Deep / Wide Research:** For search-oriented tasks.
*   **Webpage Generation:** Creating web pages based on specific inputs.
*   **Trajectory Collection:** Supporting data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides value to:

### For Agents Researchers & LLM Trainers

*   Provides a **powerful baseline** for model training and ablation studies.
*   Offers **one-click evaluation scripts** for streamlined experimentation.

### For Agent Application Developers

*   Offers a **proven and portable foundation** for real-world agent apps.
*   Features **ease of use** through simple scripts and built-in toolkits.
*   Offers a **modular design** for customizing key components.

### For AI & Agent Enthusiasts

*   Includes **practical use cases** in the `/examples` directory.
*   Features **simplicity and debuggability** with a rich toolset and tracing tools.

## 🧩 Core Concepts

*   **Agent:** An LLM with specific prompts, tools, and environment.
*   **Toolkit:** A set of tools for an agent to use.
*   **Environment:** The operating context of the agent.
*   **ContextManager:** A module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for specific datasets.

For detailed design and implementation information, refer to our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Follow these steps to quickly set up and run your first Youtu-Agent.  For a Docker-based setup, refer to [`docker/README.md`](./docker/README.md).

### Setup

#### Source Code Deployment

> [!NOTE]
> Requires Python 3.12+ and recommends [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Install Python and uv:**  Ensure Python and uv are installed.
2.  **Clone the repository:**

    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```

3.  **Sync dependencies:**

    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure API keys in .env
    ```

    Fill in the necessary keys in `.env`, such as:

    ```bash
    # llm requires OpenAI API format compatibility
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    OR, use the Tencent Cloud DeepSeek API with **3 million free tokens** (**Sep 1 – Oct 31, 2025**):

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup.

### Quick Start

Youtu-Agent includes built-in configurations. For example, `configs/agents/simple/base_search.yaml`:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Run the CLI chatbot:

```bash
# Configure SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (Alternatives will be available in the future)
python scripts/cli_chat.py --stream --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config simple/base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in `.env` for the following examples:

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example: Generate an SVG image on “DeepSeek V3.1 New Features”:

```bash
python examples/svg_generator/main.py
```

Run the web UI version:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.2.0/utu_agent_ui-0.2.0-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.2.0-py3-none-any.whl
```

Then run:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the displayed local link (e.g., `http://127.0.0.1:8848/`).

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

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 📖 Dive Deeper

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**: Explore the core concepts, architecture, and advanced features.
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**: A detailed guide to get you up and running.
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**: Find answers to common questions and issues.

## 🙏 Acknowledgements

Thanks to the open-source projects that made this possible:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 🙌 Contributing

Join the community!  See the [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute.

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