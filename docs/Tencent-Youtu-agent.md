# Youtu-Agent: Build and Deploy Powerful Autonomous Agents with Open-Source Models

Youtu-Agent empowers developers to create high-performance, cost-effective autonomous agents, delivering robust capabilities with the power of open-source models.  [Explore the Youtu-Agent Repository](https://github.com/Tencent/Youtu-agent)

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

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

Youtu-Agent is a flexible and performant framework designed for building, running, and evaluating autonomous agents, offering a compelling alternative to proprietary solutions with its open-source focus.  It excels in tasks like data analysis, file processing, and in-depth research.

**Key Features:**

*   **Impressive Performance:** Achieves strong benchmark results (e.g., 71.47% on WebWalkerQA, 72.8% on GAIA text-only subset) using open-source `DeepSeek-V3` models.
*   **Cost-Effective & Open-Source Focused:** Optimized for deployment with accessible, low-cost models, reducing reliance on expensive closed-source alternatives.
*   **Versatile Use Cases:** Provides out-of-the-box support for data analysis, file organization, literature review, and more, with upcoming support for podcast/video generation.
*   **Flexible and Extensible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting various model APIs (DeepSeek, gpt-oss, etc.), tool integrations, and framework implementations.
*   **Simplified Development:** YAML-based configurations, automated agent generation, and streamlined setup reduce manual effort and accelerate development.

## 🗞️ News

*   🎁 [2025-09-02] [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free if you want to use DeepSeek models in `Youtu-Agent`! For enterprise agent solutions, also check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   📺 [2025-08-28] We made a live sharing updates about DeepSeek-V3.1 and how to use it in the `Youtu-Agent` framework. We share the used [documentations](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent leverages open-source models and lightweight tools, demonstrating excellent performance on challenging benchmarks.

*   **WebWalkerQA:** Achieved 71.47% accuracy with `DeepSeek-V3.1`, setting a new state-of-the-art (SOTA).
*   **GAIA:** Achieved 72.8% pass@1 on the text-only validation subset using `DeepSeek-V3-0324`.  Further evaluation is underway for the full GAIA benchmark.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Click the images below for video demonstrations of Youtu-Agent in action:

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
      <strong>Wide Research</strong><br>Generates a comprehensive report.
      <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" 
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height=300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Parses and analyzes research papers.
        <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" 
             poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="300"
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with its automatic agent generation feature, streamlining the development process with YAML-based configurations.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Define requirements interactively and automatically generate & run agent configurations.
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

Explore more examples in the [`examples`](./examples) directory and dive deeper with the documentation at [`docs/examples.md`](./docs/examples.md).

## ✨ Features

![features](docs/assets/images/header.png)

### Design Philosophy

*   **Minimal Design:**  Focuses on simplicity and ease of use to avoid unnecessary complexity.
*   **Modular & Configurable:** Enables flexible customization and easy integration of new components.
*   **Open-Source & Cost-Conscious:** Prioritizes accessibility and cost-effectiveness for diverse applications.

### Core Features

*   **Built on openai-agents:** Leverages [openai-agents](https://github.com/openai/openai-agents-python) SDK for features like streaming, tracing, and compatibility with both `responses` and `chat.completions` APIs.  This ensures seamless integration with a wide array of models, including [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:**  Supports high-performance execution, crucial for efficient benchmarking and complex tasks.
*   **Tracing & Analysis:**  Provides a comprehensive `DBTracingProcessor` system for in-depth analysis of tool calls and agent behavior. (Coming soon).

### Automation

*   **YAML Configuration:** Simplifies agent management with structured, easily-editable configurations.
*   **Automatic Agent Generation:**  Automatically creates agent configurations based on user-defined requirements.
*   **Tooling Automation:** Includes tool evaluation, automated optimization, and custom tool generation (coming soon).

### Use Cases

*   **In-Depth Research:** Performs comprehensive research tasks.
*   **Webpage Generation:**  Creates web pages from specific inputs.
*   **Trajectory Collection:** Supports data collection for training and research.

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent delivers value to different user groups:

### For Agents Researchers & LLM Trainers

*   Provides a **strong, simplified baseline** that surpasses basic ReAct, serving as an excellent starting point for model training and ablation studies.
*   Includes **one-click evaluation scripts** to simplify and standardize the experimental process, ensuring consistent benchmarking.

### For Agent Application Developers

*   Offers a **proven and portable foundation** for creating real-world agent applications.
*   **Ease of Use:** Accelerate development with straightforward scripts and a rich toolkit.
*   **Modular Design:**  Offers customizable, encapsulated components like `Environment` and `ContextManager`.

### For AI & Agent Enthusiasts

*   **Practical Examples:**  Includes examples in the `/examples` directory like deep research report generation, data analysis, and file organization.
*   **Simplicity & Debuggability:** Utilizes a rich toolkit and visual tracing tools, making development and debugging intuitive.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** A curated collection of tools an agent can use.
*   **Environment:** The context where an agent operates (e.g., web browser, shell).
*   **ContextManager:** A customizable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, incorporating preprocessing, rollout, and judging logic.

For in-depth implementation details, consult our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent provides complete code and examples for rapid setup.  Follow these steps to launch your first agent, or explore [`docker/README.md`](./docker/README.md) for a streamlined Docker-based setup with an interactive frontend.

### Setup

#### Source Code Deployment

> [!NOTE]
>  Requires Python 3.12+. Use [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Ensure Python and uv are installed.
2.  Clone the repository:
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```
3.  Sync dependencies:
    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```
4.  Configure your environment:
    ```bash
    cp .env.example .env  # Configure API keys in .env
    ```

    After copying `.env.example`, populate the `.env` file with necessary API keys.  For example, for DeepSeek models:

    ```bash
    # llm - DeepSeek configuration
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    >   [Tencent Cloud International](https://www.tencentcloud.com/) offers new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381) for free. Replace your API key in `.env`:

    ```bash
    # llm - Tencent Cloud DeepSeek Configuration
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for a Docker-based setup with an interactive frontend.

### Quick Start

Youtu-agent includes built-in configurations. For a simple agent with a search tool, use the default configuration:

```yaml
defaults:
  - /model/base
  - /tools/search@toolkits.search
  - _self_

agent:
  name: simple-tool-agent
  instructions: "You are a helpful assistant that can search the web."
```

Launch an interactive CLI chatbot with this agent:

```bash
# Set SERPER_API_KEY and JINA_API_KEY in .env for web search.
# (Planned:  Replace with free alternatives)
python scripts/cli_chat.py --stream --config default
# Without search toolkit:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Examples requiring internet search functionality need the tool APIs configured in the `.env` file.

```bash
# tools
# serper api key, ref https://serper.dev/playground
SERPER_API_KEY=<Access the URL in the comments to get the API Key>
# jina api key, ref https://jina.ai/reader
JINA_API_KEY=<Access the URL in the comments to get the API Key>
```

Example:  Generate an SVG image on "DeepSeek V3.1 New Features":

```bash
python examples/svg_generator/main.py
```

Or, generate an SVG infographic on a research topic:

```bash
python examples/svg_generator/main_web.py
```

For web UI visualization, download and install the frontend package:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```

Then, run the web SVG generation command:

```bash
python examples/svg_generator/main_web.py
```

Access the project via the local link shown in the terminal.

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

The agent will automatically search, gather information, and output an SVG visualization.

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Youtu-Agent supports benchmarking on standard datasets, such as WebWalkerQA:

```bash
# prepare dataset
python scripts/data/process_web_walker_qa.py
# run evaluation with config ww.yaml with your custom exp_id
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

Results are stored and can be analyzed in the evaluation platform.

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

## 🙏 Acknowledgements

This project builds upon the contributions of:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

If you find this work useful, cite:

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