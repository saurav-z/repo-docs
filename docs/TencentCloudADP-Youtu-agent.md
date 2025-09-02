# Youtu-agent: Build Powerful Agents with Open-Source Models

<div align="center">
<a href="https://tencentcloudadp.github.io/Youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/TencentCloudADP/Youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
</div>

[![Star History](https://api.star-history.com/svg?repos=TencentCloudADP/Youtu-agent&type=Date)](https://star-history.com/#TencentCloudADP/Youtu-agent)

**Youtu-agent empowers developers to create versatile and efficient autonomous agents, offering strong performance with the accessibility of open-source models.**

<img src="docs/assets/mascot.png" alt="Youtu-agent Logo" width="200" align="left" style="margin-right:20px;">

**Key Features:**

*   ✅ **High Performance:** Achieves impressive results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8%) using open-source DeepSeek models.
*   💡 **Open-Source & Cost-Effective:** Designed for accessible, low-cost deployment, eliminating reliance on proprietary models.
*   🚀 **Practical Applications:** Supports diverse tasks including data analysis, file management, and research, with more functionalities on the way.
*   ⚙️ **Flexible Architecture:** Built on [openai-agents](https://github.com/openai/openai-agents-python), supporting various model APIs, tool integrations, and framework implementations.
*   🤖 **Automated Agent Generation:** YAML-based configurations, automatic agent creation, and simplified setup streamline development.

## Core Capabilities:

*   **Open-Source Model Support:** Leverages models like DeepSeek for high performance.
*   **Comprehensive Toolkits:**  Integrates essential tools for web search, file management, and more.
*   **YAML-based Configuration:** Simplifies agent setup and management.
*   **Automated Agent Generation:**  Streamlines the creation of custom agent configurations based on your requirements.
*   **Built-in Examples:**  Ready-to-use examples to get you started quickly.

## Use Cases & Applications:

*   **Data Analysis:** Process and analyze CSV files, generating insightful reports.
*   **File Management:** Organize and categorize files with automated renaming and sorting.
*   **Research & Information Gathering:**  Compile comprehensive reports from the web, and analyze documents.
*   **Web Page Generation:** Automate the generation of informative web pages.

## 🌟 Benchmark Performance

`Youtu-agent` leverages open-source models and lightweight tools, achieving strong results on deep search and tool use benchmarks.

*   **[WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA)**: Achieved 71.47% accuracy with `DeepSeek-V3.1`.
*   **[GAIA](https://gaia-benchmark-leaderboard.hf.space/)**: Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

## 💡 Examples

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

### 🤖 Automatic Agent Generation

Youtu-agent simplifies agent creation with **automatic configuration generation**. Just specify your needs, and the framework will handle the setup.

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

## 🚀 Getting Started

### Prerequisites

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) (Recommended)

### Installation
```bash
git clone https://github.com/TencentCloudADP/Youtu-agent.git
cd Youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # config necessary keys...
```

### Quickstart

Run a simple agent with the built-in configuration:

```bash
python scripts/cli_chat.py --stream --config default
```

### Explore examples
Generate an SVG infographic based on a research topic:

```bash
python examples/svg_generator/main_web.py
```

### Run evaluations

Evaluate on the WebWalkerQA dataset:

```bash
# prepare dataset
python scripts/data/process_web_walker_qa.py
# run evaluation with config ww.yaml with your custom exp_id
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

## 🙏 Acknowledgements

This project is built upon the foundation of these open-source projects:

*   [openai-agents](https://github.com/openai/openai-agents-python)
*   [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
*   [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

## 📚 Citation

```bibtex
@misc{youtu-agent-2025,
  title={Youtu-agent: A Simple yet Powerful Agent Framework},
  author={Tencent Youtu Lab},
  year={2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/TencentCloudADP/Youtu-agent}},
}
```

**For more detailed information, including API documentation and advanced use cases, explore the full documentation on the [Youtu-agent GitHub repository](https://github.com/TencentCloudADP/Youtu-agent).**