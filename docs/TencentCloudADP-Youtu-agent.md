# Youtu-Agent: Build Powerful Agents with Open-Source Models

**[Youtu-Agent](https://github.com/TencentCloudADP/Youtu-agent) is a flexible and high-performance agent framework enabling you to create, run, and evaluate intelligent agents using open-source LLMs.**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

---

## Key Features

*   **High-Performance & Optimized:** Achieve strong results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1, text-only) with open-source DeepSeek-V3 models.
*   **Cost-Effective & Open-Source Focused:** Designed for accessible, low-cost deployment, avoiding reliance on closed-source models.
*   **Versatile Use Cases:** Supports data analysis, file processing, research, and upcoming features like podcast and video generation.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting various model APIs, tool integrations, and framework implementations.
*   **Simplified Automation:** YAML-based configuration, automatic agent generation, and streamlined setup minimize manual effort.

---

## Why Choose Youtu-Agent?

*   **For Researchers:** Use it as a powerful baseline for model training and ablation studies, along with one-click evaluation scripts.
*   **For Developers:** A ready-to-use framework for building real-world agent applications.
*   **For Enthusiasts:** Get started quickly with examples and simplified development and debugging tools.

---

## Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** A collection of tools an agent can use.
*   **Environment:** The setting the agent operates in (e.g., a browser or shell).
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** A defined workflow for assessing performance.

---

## Getting Started

### Prerequisites

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (recommended)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```
2.  Install dependencies:
    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```
3.  Configure your `.env` file with necessary API keys:

    ```bash
    # llm
    # setup your LLM config , ref https://api-docs.deepseek.com/
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-chat
    UTU_LLM_BASE_URL=https://api.deepseek.com/v1
    UTU_LLM_API_KEY=replace-to-your-api-key
    ```

    Or, use the Tencent Cloud DeepSeek API, with free tokens available:

    ```bash
    # llm
    # setup your LLM config , ref https://www.tencentcloud.com/document/product/1255/70381
    UTU_LLM_TYPE=chat.completions
    UTU_LLM_MODEL=deepseek-v3
    UTU_LLM_BASE_URL=https://api.lkeap.cloud.tencent.com/v1
    UTU_LLM_API_KEY=replace-with-your-api-key
    ```

### Quickstart

Run a simple agent with a search tool:

```bash
# NOTE: You need to set `SERPER_API_KEY` and `JINA_API_KEY` in `.env` for web search access.
# (We plan to replace these with free alternatives in the future)
python scripts/cli_chat.py --config simple/base_search
# To avoid using the search toolkit, you can run:
python scripts/cli_chat.py --config simple/base
```

See the [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart) for more details.

### Examples

Explore practical examples, such as SVG image generation, to visualize agent behavior.  You may need to configure tool APIs in the `.env` file.

```bash
python examples/svg_generator/main.py
```
For web UI, see instructions in the original README.

### Run Evaluations

Evaluate on WebWalkerQA (after preparing the dataset):

```bash
python scripts/data/process_web_walker_qa.py
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5
```

See [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval) for results analysis.

---

## Deep Dive

*   📖 **[Full Documentation](https://tencentcloudadp.github.io/youtu-agent/)**
*   🚀 **[Quickstart Guide](https://tencentcloudadp.github.io/youtu-agent/quickstart/)**
*   ❓ **[FAQ](https://tencentcloudadp.github.io/youtu-agent/faq)**

---

## Acknowledgements

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

---

## Contribute

Read the [**Contributing Guidelines**](./CONTRIBUTING.md) to contribute to Youtu-Agent.

---

## Citation

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

---

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)