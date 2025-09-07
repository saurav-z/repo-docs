# Youtu-Agent: Build Powerful AI Agents with Open-Source Models

**Youtu-Agent is a flexible and high-performance framework that empowers you to build, run, and evaluate autonomous AI agents using open-source models.** [Explore the Youtu-Agent Repo](https://github.com/TencentCloudADP/Youtu-agent)

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

---

## Key Features

*   **Exceptional Performance**: Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47% pass@1) and GAIA (72.8% pass@1, text-only subset) using open-source models (DeepSeek-V3 series).
*   **Cost-Effective and Open-Source Focused**: Optimized for accessible, low-cost deployment, avoiding reliance on proprietary models.
*   **Versatile Use Cases**: Supports data analysis, file processing, research, and more with ready-to-use examples.
*   **Flexible and Extensible Architecture**: Built on [openai-agents](https://github.com/openai/openai-agents-python) and easily adaptable to diverse model APIs and tool integrations.
*   **Simplified Development**: Includes YAML-based configuration, automated agent generation, and streamlined setup for efficient development.

## What's New

*   **Free DeepSeek API Tokens**: New users of the DeepSeek API can receive 3 million free tokens from Tencent Cloud International (Sep 1 – Oct 31, 2025).
*   **DeepSeek-V3.1 Update**:  Check out the latest updates on DeepSeek-V3.1 and its integration with Youtu-Agent.  [View the documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## Benchmark Performance

Youtu-Agent showcases impressive performance on challenging benchmarks using open-source models:

*   **WebWalkerQA**: Achieved 71.47% accuracy using DeepSeek-V3.1, setting a new SOTA.
*   **GAIA**: Achieved 72.8% pass@1 on the text-only subset using DeepSeek-V3-0324.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## Examples

**Explore the power of Youtu-Agent with these interactive examples:**

| Data Analysis                             | File Management                        |
| :---------------------------------------- | :------------------------------------- |
| <video src="https://github.com/user-attachments/assets/b6aba820-368e-427f-ba71-85543a751775" poster="https://img.youtube.com/vi/SCR4Ru8_h5Q/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/dbb9cfc6-3963-4264-ba93-9ba21c5a579e" poster="https://img.youtube.com/vi/GdA4AapE2L4/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |
| Wide Research                             | Paper Analysis                           |
| <video src="https://github.com/user-attachments/assets/6fc75814-e565-4f94-9ab5-33e3e7788e92" poster="https://img.youtube.com/vi/v3QQg0WAnPs/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> | <video src="https://github.com/user-attachments/assets/09b24f94-30f0-4e88-9aaf-9f3bbf82e99d" poster="https://img.youtube.com/vi/vBddCjjRk00/sddefault.jpg" controls muted preload="metadata" width="100%" height="300" style="object-fit: cover; border-radius: 8px;"></video> |

### Automatic Agent Generation

Effortlessly create agent configurations with a simple YAML-based system.

| Automatic Agent Generation |
| :------------------------ |
| <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" controls muted preload="metadata" width="100%" height="auto" style="object-fit: cover; border-radius: 8px;"></video> |

For more detailed examples, explore the [`examples`](./examples) directory and comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## Architecture & Features

![features](docs/assets/images/header.png)

**Design Philosophy:**

*   **Minimal Design:** Focus on simplicity and ease of use.
*   **Modular & Configurable:** Customizable and easily extensible for new components.
*   **Open-Source & Low-Cost:**  Promotes accessibility and cost-effectiveness.

**Core Features:**

*   **Built on openai-agents:**  Leverages the foundation of [openai-agents](https://github.com/openai/openai-agents-python), ensuring compatibility with diverse model APIs.
*   **Fully Asynchronous:**  Enables high-performance and efficient execution.
*   **Tracing & Analysis System**: Provides in-depth analysis of tool calls and agent trajectories. (Coming Soon)

**Automation:**

*   **YAML-based Configuration:**  Structured and manageable agent configurations.
*   **Automatic Agent Generation:**  Generate configurations based on user requirements.
*   **Tool Generation & Optimization:** Future support for automated tool evaluation and optimization.

**Use Cases:**

*   Deep and Wide Research
*   Webpage Generation
*   Trajectory Collection

## Why Choose Youtu-Agent?

**For Researchers & LLM Trainers:**

*   A simple yet powerful baseline for model training.
*   One-click evaluation scripts for streamlined benchmarking.

**For Agent Application Developers:**

*   Proven scaffolding for real-world agent applications.
*   Ease of Use: Get started quickly with simple scripts and a rich set of built-in toolkits.
*   Modular Design

**For AI & Agent Enthusiasts:**

*   Practical Use Cases
*   Simplicity & Debuggability

## Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools.
*   **Environment:** The context in which the agent operates.
*   **ContextManager:** For managing the agent's context window.
*   **Benchmark:** Workflow for a specific dataset.

For detailed information, please consult the [technical documentation](https://tencentcloudadp.github.io/youtu-agent/).

## Getting Started

Youtu-Agent provides complete code and examples to help you get started quickly.

### Setup

Follow these steps to run your first agent:

1.  **Install Prerequisites:** Python 3.12+ and `uv` for dependency management (recommended).
2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    ```
3.  **Sync Dependencies:**
    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```
4.  **Configure API Keys:**
    ```bash
    cp .env.example .env  # Fill in your LLM API keys and tool API keys in the .env file.
    ```

    **DeepSeek API Users:** Take advantage of the free 3 million tokens offer from Tencent Cloud International (Sep 1 – Oct 31, 2025)!

### Quick Start

Use the default configuration to launch an interactive CLI chatbot:

```bash
python scripts/cli_chat.py --stream --config default
```

(Configure search tool APIs in `.env` for web search functionality).

## Explore More

Run the SVG image generation example:

```bash
python examples/svg_generator/main.py
```

Or, run the web version of the SVG image generation command:

```bash
# Download the frontend package
curl -LO https://github.com/Tencent/Youtu-agent/releases/download/frontend%2Fv0.1.5/utu_agent_ui-0.1.5-py3-none-any.whl

# Install the frontend package
uv pip install utu_agent_ui-0.1.5-py3-none-any.whl
```

```bash
python examples/svg_generator/main_web.py
```

## Run Evaluations

Evaluate on WebWalkerQA:

```bash
python scripts/data/process_web_walker_qa.py # Prepare Dataset
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA_15 --concurrency 5 # Run Evaluation
```

## Acknowledgements

We appreciate the contributions of these open-source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

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

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)