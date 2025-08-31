# 🤖 Youtu-agent: Build Powerful Agents with Open-Source Models

**Unlock the potential of autonomous agents with Youtu-agent, a flexible and high-performing framework built for efficiency and cost-effectiveness.** Explore the original repository at [https://github.com/Tencent/Youtu-agent](https://github.com/Tencent/Youtu-agent).

<div align="center">
<a href="https://tencent.github.io/Youtu-agent/"><img src=https://img.shields.io/badge/📖-Documentation-blue.svg></a>
<a href=https://github.com/Tencent/Youtu-agent><img src=https://img.shields.io/badge/GitHub-Tencent-blue.svg></a>
<a href=https://deepwiki.com/Tencent/Youtu-agent><img src=https://img.shields.io/badge/DeepWiki-Tencent-blue.svg></a>
</div>

<p align="center">
| <a href="README_ZH.md"><b>中文版</b></a>
| <a href="#benchmark-performance"><b>🌟 Performance</b></a> 
| <a href="#examples"><b>💡 Examples</b> </a> 
| <a href="#features"><b>✨ Features</b> </a> 
| <a href="#getting-started"><b>🚀 Getting Started</b> </a> 
| 
</p>

## Key Features:

*   **✅ High-Performance & Efficient:** Built on open-source models, with demonstrated performance on challenging benchmarks without reliance on closed models.
*   **🚀 Open-Source & Cost-Effective:** Optimized for accessible, low-cost deployment and supports a wide range of open-source models.
*   **💡 Practical Use Cases:** Out-of-the-box support for tasks like CSV analysis, literature review, file organization, and more.
*   **⚙️ Flexible Architecture:**  Built upon [openai-agents](https://github.com/openai/openai-agents-python), with extensive support for diverse model APIs and tool integrations.
*   **🤖 Automated Agent Generation:** YAML-based configurations and automated agent generation streamline setup and reduce manual effort.

## News

*   **[2025-08-28]** We made a live sharing updates about DeepSeek-V3.1 and how to apply it in the `Youtu-agent` framework. [Here](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF) is the documentation.

## ✨ Features

*   **Minimal Design:** Simple, easy-to-use framework avoiding unnecessary overhead.
*   **Modular & Configurable:** Flexible customization and easy integration of new components.
*   **Open-Source Model Support & Low-Cost:** Promotes accessibility and cost-effectiveness for various applications.

### Core Features
- **Built on openai-agents**: Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
- **Fully asynchronous**: Enables high-performance and efficient execution, especially beneficial for evaluating benchmarks.
- **Tracing & analysis system**: Beyond OTEL, our `DBTracingProcessor` system provides in-depth analysis of tool calls and agent trajectories. (will be released soon)

### Automation
- **YAML based configuration**: Structured and easily manageable agent configurations.
- **Automatic agent generation**: Based on user requirements, agent configurations can be automatically generated.
- **Tool generation & optimization**: Tool evaluation and automated optimization, and customized tool generation will be supported in the future.

### Use Cases
- **Deep / Wide research**: Covers common search-oriented tasks.
- **Webpage generation**: Examples include generating web pages based on specific inputs.
- **Trajectory collection**: Supports data collection for training and research purposes.

## 🌟 Benchmark Performance

`Youtu-agent` achieves strong results on deep search and tool use benchmarks using open-source models and lightweight tools.

*   **WebWalkerQA:** Achieved 71.47% pass@1 using `DeepSeek-V3.1`.
*   **GAIA (text-only subset):** Achieved 72.8% pass@1 using `DeepSeek-V3-0324`.

![WebWalkerQA](docs/assets/images/benchmark_webwalkerqa.png)

## 💡 Examples

<details>
<summary>Expand to see examples</summary>
<br>
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
</details>

### 🤖 Automatic Agent Generation

<details>
<summary>Expand to see Automatic Agent Generation</summary>
<br>
A standout feature of `Youtu-agent` is its ability to **automatically generate agent configurations**. In other frameworks, defining a task-specific agent often requires writing code or carefully crafting prompts. In contrast, `Youtu-agent` uses simple YAML-based configs, which enables streamlined automation: a built-in "meta-agent" chats with you to capture requirements, then generates and saves the config automatically.

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

</details>

For more detailed examples, refer to the [`examples`](./examples) directory and the documentation at [`docs/examples.md`](./docs/examples.md).

## 🤔 Why Choose Youtu-agent?

*   **For Agents Researchers & LLM Trainers:** A simple, powerful baseline for model training. One-click evaluation scripts streamline the experimental process.
*   **For Agent Application Developers:**  Proven scaffolding for building real-world agent applications with ease of use and modular design.
*   **For AI & Agent Enthusiasts:** Practical use cases in the `/examples` directory, with simplicity and debuggability.

## 🧩 Core Concepts

*   **Agent:** LLM configured with prompts, tools, and an environment.
*   **Toolkit:** Encapsulated set of tools.
*   **Environment:** The world the agent operates in.
*   **ContextManager:** Module for managing the agent's context window.
*   **Benchmark:** Encapsulated workflow for a dataset.

## 🚀 Getting Started

### Setup

```bash
git clone https://github.com/Tencent/Youtu-agent.git
cd Youtu-agent
uv sync  # or, `make sync`
source ./.venv/bin/activate
cp .env.example .env  # config necessary keys...
```

### Quickstart

Run a basic agent with a search tool:

```bash
python scripts/cli_chat.py --stream --config default
```

📖 More details: [Quickstart Documentation](https://tencent.github.io/Youtu-agent/quickstart)

### Explore examples

Generate an SVG infographic:

```bash
python examples/svg_generator/main_web.py
```

![svg_generator_ui](https://github.com/user-attachments/assets/337d327f-91ee-434e-bbcf-297dd4b26c28)

![svg_generator_result](https://github.com/user-attachments/assets/41aa7348-5f02-4daa-b5b2-225e35d21067)

📖 Learn more: [Examples Documentation](https://tencent.github.io/Youtu-agent/examples)

### Run evaluations

Evaluate on `WebWalkerQA`:

```bash
# prepare dataset
python scripts/data/process_web_walker_qa.py
# run evaluation with config ww.yaml with your custom exp_id
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

![eval_analysis_overview](https://github.com/user-attachments/assets/4a285b9e-d096-437e-9b8e-e5bf6b1924b6)

![eval_analysis_detail](https://github.com/user-attachments/assets/4ede525a-5e16-4d88-9ebb-01a7dca3aaec)

📖 Learn more: [Evaluation Documentation](https://tencent.github.io/Youtu-agent/eval)

## 🙏 Acknowledgements

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
  howpublished = {\url{https://github.com/Tencent/Youtu-agent}},
}
```

## ⭐ Star History

![Star History Chart](https://api.star-history.com/svg?repos=Tencent/Youtu-agent&type=Date)