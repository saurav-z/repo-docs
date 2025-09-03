# Youtu-Agent: Build Powerful Agents with Open-Source Models

**Unlock the potential of autonomous agents with Youtu-Agent, a flexible and high-performing framework that delivers cutting-edge capabilities using open-source models. Explore the [original repository](https://github.com/TencentCloudADP/Youtu-agent) for more details.**

[![Documentation](https://img.shields.io/badge/📖-Documentation-blue.svg)](https://tencentcloudadp.github.io/youtu-agent/)
[![GitHub](https://img.shields.io/badge/GitHub-Tencent-blue.svg)](https://github.com/TencentCloudADP/youtu-agent)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Tencent-blue.svg)](https://deepwiki.com/TencentCloudADP/youtu-agent)

---

## Key Features

*   **High-Performance with Open-Source:** Achieves impressive benchmark results (71.47% on WebWalkerQA and 72.8% on GAIA text-only subset) using DeepSeek-V3 models, demonstrating strong performance with cost-effective open-source options.
*   **Cost-Effective and Accessible:** Designed for deployment with open-source models, minimizing reliance on expensive closed-source alternatives.
*   **Versatile Use Cases:** Supports a variety of tasks, including:
    *   CSV analysis
    *   Literature review
    *   Personal file organization
    *   Podcast and video generation (coming soon)
*   **Flexible and Extensible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting a wide range of model APIs (DeepSeek, gpt-oss, etc.), tool integrations, and framework implementations.
*   **Automated Agent Creation:** Simplify agent development with YAML-based configurations and automated agent generation.

## 🚀 Getting Started

Youtu-Agent makes it easy to start building and running agents.

### 1. Setup

*   **Prerequisites**: Python 3.12+ is required. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.
    ```bash
    git clone https://github.com/TencentCloudADP/youtu-agent.git
    cd youtu-agent
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    cp .env.example .env  # Configure your API keys here.
    ```
    *   **Important:** Configure the necessary API keys in the `.env` file, specifically for your chosen LLM (DeepSeek, etc.).
    *   Tencent Cloud International is offering new users of the DeepSeek API **3 million free tokens** (**Sep 1 – Oct 31, 2025**). [Try it out](https://www.tencentcloud.com/document/product/1255/70381)

### 2. Quickstart

Launch an interactive CLI chatbot with a default search agent:

```bash
#  Set SERPER_API_KEY and JINA_API_KEY in `.env` for web search access.
python scripts/cli_chat.py --stream --config default
```

📖 Learn More: [Quickstart Documentation](https://tencentcloudadp.github.io/youtu-agent/quickstart)

### 3. Explore Examples

Run example agents for practical tasks:

```bash
# For example, generate an SVG image based on a research topic:
python examples/svg_generator/main.py
```

📖 Learn More: [Examples Documentation](https://tencentcloudadp.github.io/youtu-agent/examples)

### 4. Run Evaluations

Benchmark your agent on standard datasets:

```bash
# Prepare the dataset:
python scripts/data/process_web_walker_qa.py
# Run evaluation (replace <your_exp_id>):
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

📖 Learn More: [Evaluation Documentation](https://tencentcloudadp.github.io/youtu-agent/eval)

---

## 🤖 Automatic Agent Generation

Youtu-Agent simplifies agent creation with YAML-based configurations and automated generation.

### Simplified Configuration
```bash
# Interactively specify requirements to auto-generate configuration
python scripts/gen_simple_agent.py
# Run the generated configuration
python scripts/cli_chat.py --stream --config generated/xxx
```

### Demo: Automatic Agent Creation

<video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef"
       poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg"
       controls muted preload="metadata"
       width="100%" height="auto"
       style="object-fit: cover; border-radius: 8px;"></video>

---

## Benchmark Performance

Youtu-Agent demonstrates strong performance on challenging benchmarks:

*   **WebWalkerQA:** Achieved 71.47% accuracy using DeepSeek-V3.1.
*   **GAIA (text-only subset):** Achieved 72.8% pass@1 using DeepSeek-V3-0324.

---

## 💡 Examples

Explore practical applications of Youtu-Agent with these interactive examples.

<table style="width:100%;">
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Data Analysis</strong><br>Analyzes CSV files to generate an HTML report.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>File Management</strong><br>Organizes local files by renaming and categorizing them.
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
      <strong>Wide Research</strong><br>Generates comprehensive reports by gathering information.
    </td>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <strong>Paper Analysis</strong><br>Analyzes papers, identifies relevant literature, and compiles findings.
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

---

## 🤔 Why Choose Youtu-Agent?

Youtu-Agent is an excellent choice for:

*   **Agents Researchers & LLM Trainers:** Provides a strong baseline for model training and ablation studies.
*   **Agent Application Developers:** Offers a solid foundation for building real-world agent applications.
*   **AI & Agent Enthusiasts:** Offers practical use cases, simplicity, and debugging tools for intuitive development.

---

## 🧩 Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools an agent can use.
*   **Environment:** The world where the agent operates (e.g., browser, shell).
*   **ContextManager:**  Manages the agent's context window.
*   **Benchmark:** Workflow for a specific dataset (preprocessing, rollout, and judging).

---

## 🙏 Acknowledgements

This project is built upon the work of these great open source projects:
*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

---

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

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TencentCloudADP/youtu-agent&type=Date)](https://star-history.com/#TencentCloudADP/youtu-agent)