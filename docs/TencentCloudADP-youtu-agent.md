# 🤖 Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Unlock the potential of AI agents with Youtu-Agent, a flexible and high-performance framework designed to build, run, and evaluate agents using open-source models.  [Explore the Youtu-Agent Repository](https://github.com/TencentCloudADP/youtu-agent)**

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

Youtu-Agent empowers developers and researchers to create cutting-edge AI agents for diverse applications, leveraging the power of open-source language models and minimizing costs.

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

## ✨ Key Features

*   **Top-Tier Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%, text-only subset) using open-source models.
*   **Cost-Effective & Open-Source Focused:** Designed for efficient deployment with open-source models, reducing reliance on expensive closed-source solutions.
*   **Versatile Use Cases:** Supports data analysis, file processing, literature review, and is expanding to include podcast and video generation.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), with support for various model APIs (e.g., DeepSeek, gpt-oss), tool integrations, and framework implementations.
*   **Simplified Agent Creation:**  YAML-based configurations and automatic agent generation streamline the development process, reducing manual setup.

## 🗞️ News

*   **[2025-09-02]** [Tencent Cloud International](https://www.tencentcloud.com/) is offering new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**).  [Get started](https://www.tencentcloud.com/document/product/1255/70381) for free with DeepSeek models in `Youtu-Agent`!  For enterprise agent solutions, check out [Agent Development Platform](https://adp.tencentcloud.com) (ADP).
*   **[2025-08-28]**  Catch up on the latest updates on DeepSeek-V3.1 and its integration with `Youtu-Agent` in the [documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## 🌟 Benchmark Performance

Youtu-Agent demonstrates exceptional performance on challenging benchmarks using open-source models and lightweight tools.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**:  Achieved 71.47% accuracy with `DeepSeek-V3.1`, a new state-of-the-art result.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.  Multimodal tool evaluation is ongoing.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

Explore practical applications of Youtu-Agent through these interactive examples:

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
      <strong>File Management</strong><br>Renames and categorizes local files for the user.
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
      <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" 
             poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height=300"
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

### 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent development with its ability to automatically generate agent configurations using intuitive YAML-based configs.

```bash
# Interactively clarify your requirements and auto-generate a config
python scripts/gen_simple_agent.py

# Run the generated config
python scripts/cli_chat.py --stream --config generated/xxx
```

<table>
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Automatic Agent Generation</strong><br>Interactively define your needs and automatically generate and execute agent configurations.
      <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" 
             poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="auto" 
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

For in-depth examples and advanced usage, explore the [`examples`](./examples) directory and the detailed documentation at [`docs/examples.md`](./docs/examples.md).

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent provides significant advantages for:

### For Agents Researchers & LLM Trainers
*   **Strong Baseline:** Offers a powerful starting point, surpassing basic ReAct, for model training and ablation studies.
*   **Simplified Evaluation:** Includes one-click evaluation scripts for consistent benchmarking.

### For Agent Application Developers
*   **Practical Scaffolding:**  Provides a reliable framework for creating real-world agent applications.
*   **User-Friendly:**  Offers easy setup with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:** Key components such as `Environment` and `ContextManager` are encapsulated yet highly customizable.

### For AI & Agent Enthusiasts
*   **Practical Applications:** Explore diverse use cases like deep research report generation and data analysis in the `/examples` directory.
*   **Simplified Debugging:** A robust toolkit and visual tracing tools make development and debugging streamlined.

## 🧩 Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools accessible to an agent.
*   **Environment:** The operational context of the agent (e.g., a browser, a shell).
*   **ContextManager:** A configurable module managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for specific datasets including preprocessing, rollout, and judging logic.

Learn more about design and implementation details in our [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## 🚀 Getting Started

Youtu-Agent is easy to set up with provided code and examples.

### Setup

#### Source Code Deployment

> [!NOTE]
> The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

1.  Ensure Python and uv are installed.

2.  Clone the repository and install dependencies:

```bash
git clone https://github.com/TencentCloudADP/youtu-agent.git
cd youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # NOTE: You should then config the necessary API keys.
```

3.  Configure the `.env` file with your API keys, such as LLM API keys, after copying the `.env.example` file. For example:

```bash
# llm requires OpenAI API format compatibility
# setup your LLM config , ref https://api-docs.deepseek.com/
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-chat
UTU_LLM_BASE_URL=https://api.deepseek.com/v1
UTU_LLM_API_KEY=replace-to-your-api-key
```

4.  [Tencent Cloud International](https://www.tencentcloud.com/) is offering new DeepSeek API users **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Get started](https://www.tencentcloud.com/document/product/1255/70381) for free.  Once you’ve applied, replace the API key in the .env file:

```bash
# llm
# setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=deepseek-v3
UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
UTU_LLM_API_KEY=replace-with-your-api-key
```

#### Docker Deployment

Refer to [`docker/README.md`](./docker/README.md) for Docker-based setup with an interactive frontend.

### Quick Start

Youtu-agent provides built-in configurations. For example, the default config (`configs/agents/default.yaml`) defines a simple agent equipped with a search tool:

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
# NOTE: Set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --stream --config default
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --stream --config base
```

📖 More details: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### Explore More Examples

Configure tool APIs in the `.env` file:

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

Run the web version of the SVG image generation command:

```bash
python examples/svg_generator/main_web.py
```

Access the project at http://127.0.0.1:8848/

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### Run Evaluations

Evaluate on `WebWalkerQA`:

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

This project is built on the foundations of:
-   [openai-agents](https://github.com/openai/openai-agents-python)
-   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
-   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

If you find this work useful, please cite:

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