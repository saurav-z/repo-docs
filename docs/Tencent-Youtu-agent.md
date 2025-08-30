# Youtu-agent: Build Powerful Agents with Open-Source Models

**Unleash the power of autonomous agents with Youtu-agent, a high-performance framework designed for building, running, and evaluating agents using open-source models.  [Explore the Youtu-agent Repository](https://github.com/Tencent/Youtu-agent)**

Youtu-agent empowers you to create sophisticated agents for data analysis, file processing, and in-depth research, all while leveraging the cost-effectiveness and flexibility of open-source models.

**Key Features:**

*   **Impressive Performance:** Achieves state-of-the-art results on benchmarks like WebWalkerQA (71.47%) and GAIA (72.8% text-only), demonstrating strong capabilities with open-source models.
*   **Open-Source & Cost-Effective:** Designed for accessible deployment, eliminating reliance on expensive closed-source models.
*   **Practical Use Cases:** Supports tasks like CSV analysis, literature review, personal file management, and more, with podcast and video generation coming soon.
*   **Flexible Architecture:** Built on the robust [openai-agents](https://github.com/openai/openai-agents-python), offering extensive support for diverse model APIs (DeepSeek, gpt-oss), tool integrations, and framework implementations.
*   **Streamlined Automation:** YAML-based configuration, automatic agent generation, and simplified setup minimize manual effort.

## Key Highlights:

*   **Verified Performance:** Achieved 71.47% on WebWalkerQA (pass@1) and 72.8% on GAIA (text-only subset, pass@1), using purely `DeepSeek-V3` series models (without Claude or GPT), establishing a strong open-source starting point.
*   **Open-source friendly & cost-aware**: Optimized for accessible, low-cost deployment without reliance on closed models.
*   **Practical use cases**: Out-of-the-box support for tasks like CSV analysis, literature review, personal file organization, and podcast and video generation (coming soon).
*   **Flexible architecture**: Built on [openai-agents](https://github.com/openai/openai-agents-python), with extensible support for diverse model APIs (form `DeepSeek` to `gpt-oss`), tool integrations, and framework implementations.
*   **Automation & simplicity**: YAML-based configs, auto agent generation, and streamlined setup reduce manual overhead.

## News

*   **[2025-08-28]** DeepSeek-V3.1 and its application in the `Youtu-agent` framework. [Documentation](https://doc.weixin.qq.com/doc/w3_AcMATAZtAPICNvcLaY5FvTOuo7MwF).

## Performance

Youtu-agent demonstrates strong results on challenging deep search and tool use benchmarks, built upon open-source models and lightweight tools.

*   **WebWalkerQA:** Achieved 60.71% accuracy with `DeepSeek-V3-0324` and 71.47% with `DeepSeek-V3.1`.
*   **GAIA:** Achieved 72.8% pass@1 on the [text-only validation subset](https://github.com/sunnynexus/WebThinker/blob/main/data/GAIA/dev.json) using `DeepSeek-V3-0324`.

## Examples

Explore the practical applications of Youtu-agent through these interactive examples:

| Task                 | Description                                     |
| -------------------- | ----------------------------------------------- |
| Data Analysis        | Analyze CSV files and generate HTML reports.    |
| File Management      | Rename and categorize local files automatically. |
| Wide Research        | Generate comprehensive reports using extensive data. |
| Paper Analysis       | Analyze papers, compile related literature.      |

 *(Include video previews here - replace placeholders with actual video embeds)*

## Automatic Agent Generation

Youtu-agent simplifies agent creation with automatic configuration generation. Interact with a built-in "meta-agent" via YAML-based configurations to define your agent and generate config automatically.

**How it Works:**

1.  Clarify your requirements interactively.
2.  The system automatically generates the agent configuration.
3.  Run the generated configuration immediately.

 *(Include video preview here - replace placeholders with actual video embeds)*

For more detailed examples and advanced use-cases, please refer to the [`examples`](./examples) directory and our comprehensive documentation at [`docs/examples.md`](./docs/examples.md).

## Features

### Design Philosophy

*   **Minimal Design:** Simple and user-friendly, avoiding unnecessary complexity.
*   **Modular & Configurable:** Easy to customize and integrate new components.
*   **Open-Source Model Support & Low-Cost:** Promoting accessibility and cost-effectiveness.

### Core Features

*   **Based on openai-agents:** Leveraging the foundation of [openai-agents](https://github.com/openai/openai-agents-python) SDK, our framework inherits streaming, tracing, and agent-loop capabilities, ensuring compatibility with both `responses` and `chat.completions` APIs for seamless adaptation to diverse models like [gpt-oss](https://github.com/openai/gpt-oss).
*   **Fully Asynchronous:** Enables high-performance and efficient execution.
*   **Tracing & Analysis System:** The `DBTracingProcessor` provides in-depth analysis of tool calls and agent trajectories.

### Automation

*   **YAML Based Configuration:** Structured agent configurations.
*   **Automatic Agent Generation:** Automated agent configuration based on user requirements.
*   **Tool Generation & Optimization:** Tool evaluation, automated optimization, and customized tool generation.

### Use Cases

*   Deep/Wide research
*   Webpage generation
*   Trajectory collection

## Why Choose Youtu-agent?

Youtu-agent provides significant value for:

### Agent Researchers & LLM Trainers

*   **A Simple Baseline:** Serves as an excellent starting point for model training and ablation studies.
*   **One-Click Evaluation Scripts:** Streamlines the experimental process and ensures consistent benchmarking.

### Agent Application Developers

*   **A Proven Scaffolding:** For building real-world agent applications.
*   **Ease of Use:** Get started quickly with simple scripts and a rich set of built-in toolkits.
*   **Modular Design:** Key components like `Environment` and `ContextManager` are encapsulated yet highly customizable.

### AI & Agent Enthusiasts

*   **Practical Use Cases:** Examples in the `/examples` directory, including tasks like deep research report generation, data analysis, and personal file organization.
*   **Simplicity & Debuggability:** A rich toolset and visual tracing tools make development and debugging intuitive and straightforward.

## Core Concepts

*   **Agent:** An LLM configured with specific prompts, tools, and an environment.
*   **Toolkit:** An encapsulated set of tools that an agent can use.
*   **Environment:** The world in which the agent operates (e.g., a browser, a shell).
*   **ContextManager:** A configurable module for managing the agent's context window.
*   **Benchmark:** An encapsulated workflow for a specific dataset, including preprocessing, rollout, and judging logic.

For more design and implementation details, please refer to our [technical documentation](https://tencent.github.io/Youtu-agent/).

## Getting Started

Follow these steps to quickly run your first agent:

### Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/Tencent/Youtu-agent.git
    cd Youtu-agent
    ```
2.  Install dependencies:
    ```bash
    uv sync  # or, `make sync`
    source ./.venv/bin/activate
    ```
3.  Configure your environment:
    ```bash
    cp .env.example .env  # config necessary keys...
    ```

    > [!NOTE]
    > The project requires Python 3.12+. We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.

### Quickstart

Launch an interactive CLI chatbot using the default configuration:

```bash
python scripts/cli_chat.py --stream --config default
```

📖 More details: [Quickstart Documentation](https://tencent.github.io/Youtu-agent/quickstart)

### Explore Examples

Run an example to generate an SVG infographic:

```bash
python examples/svg_generator/main_web.py
```

> [!NOTE]
> To use the WebUI, you need to install the `utu_agent_ui` package. Refer to [documentation](https://tencent.github.io/Youtu-agent/frontend/#installation) for more details.

 *(Include image or video of the SVG generation here)*

📖 Learn more: [Examples Documentation](https://tencent.github.io/Youtu-agent/examples)

### Run Evaluations

Benchmark on standard datasets, such as `WebWalkerQA`:

```bash
# prepare dataset
python scripts/data/process_web_walker_qa.py
# run evaluation with config ww.yaml with your custom exp_id
python scripts/run_eval.py --config_name ww --exp_id <your_exp_id> --dataset WebWalkerQA --concurrency 5
```

 *(Include screenshots of evaluation results)*

📖 Learn more: [Evaluation Documentation](https://tencent.github.io/Youtu-agent/eval)

## Acknowledgements

This project builds upon the excellent work of several open-source projects:
- [openai-agents](https://github.com/openai/openai-agents-python)
- [mkdocs-material](https://github.com/squidfunk/mkdocs-material)
- [model-context-protocol](https://github.com/modelcontextprotocol/python-sdk)

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

## ⭐ Star History

*(Include Star History Chart)*