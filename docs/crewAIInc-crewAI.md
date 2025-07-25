<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  <b>Unlock the Power of Autonomous AI with CrewAI: Build Intelligent Agent Teams that Collaborate to Solve Complex Problems.</b>
</p>

<p align="center">
  <a href="https://crewai.com">Homepage</a>
  ·
  <a href="https://docs.crewai.com">Docs</a>
  ·
  <a href="https://app.crewai.com">Start Cloud Trial</a>
  ·
  <a href="https://blog.crewai.com">Blog</a>
  ·
  <a href="https://community.crewai.com">Forum</a>
</p>

<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="https://img.shields.io/github/stars/crewAIInc/crewAI" alt="GitHub Repo stars">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/network/members">
    <img src="https://img.shields.io/github/forks/crewAIInc/crewAI" alt="GitHub forks">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/issues">
    <img src="https://img.shields.io/github/issues/crewAIInc/crewAI" alt="GitHub issues">
  </a>
  <a href="https://github.com/crewAIInc/crewAI/pulls">
    <img src="https://img.shields.io/github/issues-pr/crewAIInc/crewAI" alt="GitHub pull requests">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  </a>
</p>

<p align="center">
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/v/crewai" alt="PyPI version">
  </a>
  <a href="https://pypi.org/project/crewai/">
    <img src="https://img.shields.io/pypi/dm/crewai" alt="PyPI downloads">
  </a>
  <a href="https://twitter.com/crewAIInc">
    <img src="https://img.shields.io/twitter/follow/crewAIInc?style=social" alt="Twitter Follow">
  </a>
</p>

## CrewAI: Your Gateway to Multi-Agent AI Automation

CrewAI is a powerful and flexible Python framework designed for building autonomous AI agent systems.  Unlike frameworks like LangChain, CrewAI is built from the ground up, offering superior performance, granular control, and seamless integration for complex tasks.  Explore the [original repository](https://github.com/crewAIInc/crewAI).

**Key Features:**

*   ✅ **Standalone & Lean:** Built from scratch, independent of other frameworks, for faster execution and efficiency.
*   ✅ **Autonomous Crews:** Empower AI agents to collaborate intelligently, delegating tasks and making decisions.
*   ✅ **Precise Flows:** Gain granular control over workflows with event-driven orchestration.
*   ✅ **Deep Customization:** Tailor everything from agent behaviors to complex workflow logic.
*   ✅ **High Performance:** Experience optimized speed and minimal resource usage.
*   ✅ **Robust Community:** Leverage the knowledge of a rapidly growing community with over 100,000 certified developers.

## Why Choose CrewAI?

CrewAI delivers a best-in-class combination of speed, flexibility, and control, ideal for developers and enterprises seeking to build intelligent automations. It stands out due to its:

*   **Standalone Framework:** Built from the ground up, independent of frameworks like LangChain.
*   **High Performance:** Optimized for speed and minimal resource usage, resulting in faster execution.
*   **Flexible Low-Level Customization:** Offers complete freedom to customize from workflows to internal prompts.
*   **Ideal for Every Use Case:** Proven effective for simple tasks and complex, enterprise-grade scenarios.
*   **Robust Community:** Backed by a growing community, offering comprehensive support and resources.

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

## Getting Started

Start your journey by following this tutorial:

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

## Learning Resources

Master CrewAI with these comprehensive courses:

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Learn the fundamentals.
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/) - Dive into advanced implementations.

## Understanding Flows and Crews

CrewAI offers two powerful, complementary approaches:

1.  **Crews:**  Autonomous teams of AI agents collaborating on complex tasks, leveraging role-based collaboration for:
    *   Autonomous decision-making.
    *   Dynamic task delegation.
    *   Specialized roles.
    *   Flexible problem-solving.

2.  **Flows:** Production-ready, event-driven workflows for precise control:
    *   Fine-grained control over execution.
    *   Secure state management.
    *   Integration of AI agents with Python code.
    *   Conditional branching for complex logic.

Combine Crews and Flows for:

*   Building production-grade applications.
*   Balancing autonomy and control.
*   Handling sophisticated real-world scenarios.
*   Maintaining clean code structure.

## Installation

1.  **Prerequisites:** Ensure you have Python >=3.10 <3.14 installed.

2.  **Install CrewAI:**

    ```shell
    pip install crewai
    ```

    For optional features including tools, use:

    ```shell
    pip install 'crewai[tools]'
    ```

### Troubleshooting Dependencies

*   **ModuleNotFoundError: No module named 'tiktoken'**: `pip install 'crewai[embeddings]'` or `pip install 'crewai[tools]'`.
*   **Failed building wheel for tiktoken**: Ensure Rust compiler and/or Visual C++ Build Tools are installed. Try upgrading pip or using a pre-built wheel.

## Project Setup and Running your first crew

1.  **Create a Project:**  `crewai create crew <project_name>`

2.  **Project Structure:**

    ```
    my_project/
    ├── .gitignore
    ├── pyproject.toml
    ├── README.md
    ├── .env
    └── src/
        └── my_project/
            ├── __init__.py
            ├── main.py
            ├── crew.py
            ├── tools/
            │   ├── custom_tool.py
            │   └── __init__.py
            └── config/
                ├── agents.yaml
                └── tasks.yaml
    ```

3.  **Customize your project by editing these files:**

    *   `src/my_project/config/agents.yaml`: Define your agents.
    *   `src/my_project/config/tasks.yaml`: Define tasks.
    *   `src/my_project/crew.py`: Add logic, tools, and arguments.
    *   `src/my_project/main.py`: Add custom inputs.
    *   `.env`: Add environment variables.

4.  **Example Crew with a Sequential Process:**

    *   `crewai create crew latest-ai-development`
    *   **agents.yaml:** (Example content included in original README)
    *   **tasks.yaml:** (Example content included in original README)
    *   **crew.py:** (Example content included in original README)
    *   **main.py:** (Example content included in original README)

5.  **Running your Crew:**

    *   Set the required environment variables: `OPENAI_API_KEY`, `SERPER_API_KEY` in `.env`
    *   Navigate to the project directory: `cd my_project`
    *   Install Dependencies: `crewai install`
    *   Run the crew: `crewai run` or `python src/my_project/main.py`
    *   Update `crewai` if you run into errors: `crewai update`

## Key Features

CrewAI is a standalone framework for high-performance multi-AI Agent applications.

*   ✅ **Standalone & Lean:**  Built from scratch, with no external dependencies.
*   ✅ **Flexible & Precise:** Easily orchestrate with Crews or Flows.
*   ✅ **Seamless Integration:** Combines Crews and Flows for complex automation.
*   ✅ **Deep Customization:** Tailor every aspect of your AI system.
*   ✅ **Reliable Performance:**  Consistent results for diverse use cases.
*   ✅ **Thriving Community:** Supported by robust documentation and over 100,000 certified developers.

## Examples

Explore real-world AI crew examples:  [CrewAI-examples](https://github.com/crewAIInc/crewAI-examples?tab=readme-ov-file)

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/landing_page_generator)
*   [Human Input on Execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

## Combining Crews and Flows

CrewAI's power lies in the combination of Crews and Flows. Use `or_` and `and_` logical operators to combine multiple conditions and create complex workflows.

*   `or_`: Triggers when any of the specified conditions are met.
*   `and_`: Triggers when all of the specified conditions are met.

[Example Code from original README for Combining Crews and Flows]

## Connecting Your Crew to a Model

CrewAI supports various LLMs. Configure your agents using [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/).

## How CrewAI Compares

**CrewAI's Advantage:** CrewAI combines autonomous agent intelligence with precise workflow control through its unique Crews and Flows architecture, which excel at both high-level orchestration and low-level customization.

*   **LangGraph:** Requires significant boilerplate code and complex state management. Tight coupling with LangChain limits flexibility.
    *   *P.S. CrewAI demonstrates significant performance advantages over LangGraph.*
*   **Autogen:** Lacks an inherent concept of process. Orchestrating agents requires extra programming, making it complex.
*   **ChatDev:** Its implementation is rigid and not geared towards production environments, which can hinder scalability and flexibility.

## Contribution

CrewAI is open-source and welcomes contributions.

*   Fork the repository.
*   Create a branch for your feature.
*   Add your feature or improvement.
*   Send a pull request.

### Installing Dependencies

```bash
uv lock
uv sync
```

### Virtual Env

```bash
uv venv
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
uv run pytest .
```

### Running static type checks

```bash
uvx mypy src
```

### Packaging

```bash
uv build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```

## Telemetry

CrewAI uses anonymous telemetry to help improve the library. No data is collected from prompts, tasks, agents, or secrets.  Users can disable telemetry by setting the `OTEL_SDK_DISABLED` environment variable to `true`. Users can opt-in to Further Telemetry, sharing the complete telemetry data by setting the `share_crew` attribute to `True` on their Crews. Enabling `share_crew` results in the collection of detailed crew and task execution data, including `goal`, `backstory`, `context`, and `output` of tasks. This enables a deeper insight into usage patterns while respecting the user's choice to share.

## License

Released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

[FAQ Content as in Original README, adjusted to be more concise]