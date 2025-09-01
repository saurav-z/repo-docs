# Youtu-agent: Build Powerful Agents with Open-Source Models

**Unlock the potential of autonomous agents with Youtu-agent, a flexible and high-performing framework that leverages the power of open-source models.** Explore the official [Youtu-agent GitHub repository](https://github.com/Tencent/Youtu-agent) for more information and to get started.

[<img src="https://img.shields.io/badge/📖-Documentation-blue.svg">](https://tencent.github.io/Youtu-agent/)
[<img src="https://img.shields.io/badge/GitHub-Tencent-blue.svg">](https://github.com/Tencent/Youtu-agent)
[<img src="https://img.shields.io/badge/DeepWiki-Tencent-blue.svg">](https://deepwiki.com/Tencent/Youtu-agent)

## Key Features

*   **High Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA and GAIA using open-source DeepSeek-V3 models.
*   **Open-Source & Cost-Effective:** Designed for accessibility, offering powerful agent capabilities without reliance on proprietary models.
*   **Versatile Use Cases:** Supports data analysis, file processing, research, and more, with pre-built examples and expanding functionality.
*   **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), enabling easy integration of diverse models, tools, and framework implementations.
*   **Automated Agent Generation:** Streamline agent creation with YAML-based configurations and a built-in "meta-agent" for automatic config generation.

## Why Choose Youtu-agent?

Youtu-agent is tailored for:

*   **Researchers & LLM Trainers:** Provides a strong baseline for experimentation and benchmarking.
*   **Agent Application Developers:** Offers a proven foundation with a rich set of toolkits and ease of use.
*   **AI & Agent Enthusiasts:** Showcases practical applications and offers a simple, debuggable framework.

## Core Concepts

*   **Agent:** An LLM configured with prompts, tools, and an environment.
*   **Toolkit:** A set of tools an agent can use.
*   **Environment:** The context in which the agent operates.
*   **ContextManager:** Manages the agent's context window.
*   **Benchmark:** A workflow for a specific dataset, including preprocessing and evaluation.

## 🚀 Getting Started

### 1. Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Tencent/Youtu-agent.git
cd Youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # config necessary keys...
```

### 2. Quickstart

Launch an interactive CLI chatbot with a default agent:

```bash
python scripts/cli_chat.py --stream --config default
```

### 3. Explore Examples

Generate an SVG infographic based on a research topic:

```bash
python examples/svg_generator/main_web.py
```

### 4. Run Evaluations

Evaluate on `WebWalkerQA`:

```bash
python scripts/data/process_web_walker_qa.py
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

## Examples

Here are some of the powerful features that you can utilize.

| Feature          | Description                                                                               |
| ---------------- | ----------------------------------------------------------------------------------------- |
| Data Analysis    | Analyzes a CSV file and generates an HTML report.                                      |
| File Management  | Renames and categorizes local files for the user.                                         |
| Wide Research    | Gathers extensive information to generate a comprehensive report, replicating Manus.      |
| Paper Analysis   | Parses a given paper, performs analysis, and compiles related literature to produce a result. |
| Auto Agent Generation | Interactively clarify your requirements, automatically generate the agent configuration, and run it right away.                                         |

<br>
<table>
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
  <tr >
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
  <tr>
    <td style="border: 1px solid black; padding: 10px; width: 50%; vertical-align: top;">
      <video src="https://github.com/user-attachments/assets/0c2ee833-507e-4141-8de4-148ff3d9f9ef" 
             poster="https://img.youtube.com/vi/JVpHDJtKBo8/maxresdefault.jpg" 
             controls muted preload="metadata" 
             width="100%" height="auto" 
             style="object-fit: cover; border-radius: 8px;"></video>
    </td>
  </tr>
</table>

## Performance Highlights

*   **WebWalkerQA:** Achieved 71.47% accuracy (pass@1) with DeepSeek-V3.1.
*   **GAIA:** Achieved 72.8% pass@1 on the text-only subset using DeepSeek-V3-0324.

## 🗞️ News

*   [2025-08-28] We made a live sharing updates about DeepSeek-V3.1 and how to apply it in the `Youtu-agent` framework. [Here](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF) is the documentation.

## Acknowledgements

This project builds upon the work of open-source projects like:
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
  howpublished = {\url{https://github.com/Tencent/Youtu-agent}},
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=Tencent/Youtu-agent&type=Date)