<p align="center">
  <a href="https://github.com/crewAIInc/crewAI">
    <img src="docs/images/crewai_logo.png" width="600px" alt="Open source Multi-AI Agent orchestration framework">
  </a>
</p>

<p align="center">
  <a href="https://trendshift.io/repositories/11239" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/11239" alt="crewAIInc%2FcrewAI | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
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

##  Supercharge Your AI Automation: Unleash the Power of Multi-Agent Collaboration with CrewAI!

CrewAI is a lightning-fast, open-source Python framework designed for building autonomous AI agents, providing unparalleled flexibility and control.  [Explore the original repo](https://github.com/crewAIInc/crewAI).

**Key Features:**

*   ✅ **Standalone & Lean:** Completely independent of frameworks like LangChain, ensuring faster execution and reduced overhead.
*   ✅ **Flexible & Precise:** Orchestrate autonomous agents effortlessly using intuitive [Crews](https://docs.crewai.com/concepts/crews) or precise [Flows](https://docs.crewai.com/concepts/flows).
*   ✅ **Seamless Integration:** Combine Crews (autonomy) and Flows (precision) for advanced, real-world automation.
*   ✅ **Deep Customization:** Tailor every aspect, from high-level workflows to low-level agent behaviors and internal prompts.
*   ✅ **Reliable Performance:** Achieve consistent results across various tasks, from simple to complex, enterprise-level automations.
*   ✅ **Thriving Community:** Benefit from extensive documentation and a rapidly growing community of over 100,000 certified developers, ensuring support and guidance.

## Table of Contents

*   [Why CrewAI?](#why-crewai)
*   [Getting Started](#getting-started)
*   [Key Features](#key-features)
*   [Examples](#examples)
    *   [Quick Tutorial](#quick-tutorial)
    *   [Write Job Descriptions](#write-job-descriptions)
    *   [Trip Planner](#trip-planner)
    *   [Stock Analysis](#stock-analysis)
    *   [Using Crews and Flows Together](#using-crews-and-flows-together)
*   [Connecting Your Crew to a Model](#connecting-your-crew-to-a-model)
*   [How CrewAI Compares](#how-crewai-compares)
*   [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
*   [Contribution](#contribution)
*   [Telemetry](#telemetry)
*   [License](#license)

## Why CrewAI?

<div align="center" style="margin-bottom: 30px;">
  <img src="docs/images/asset.png" alt="CrewAI Logo" width="100%">
</div>

CrewAI offers a best-in-class combination of speed, flexibility, and control, empowering developers and enterprises to build intelligent automations.

*   **Standalone Framework:** Built from scratch, independent of LangChain or any other agent framework.
*   **High Performance:** Optimized for speed and minimal resource usage.
*   **Flexible Customization:** Complete freedom to customize workflows, system architecture, agent behaviors, and execution logic.
*   **Ideal for Every Use Case:** Effective for simple to complex, enterprise-grade scenarios.
*   **Robust Community:** Backed by a rapidly growing community of over **100,000 certified** developers.

## Getting Started

Get up and running with CrewAI quickly with the following tutorial.

[![CrewAI Getting Started Tutorial](https://img.youtube.com/vi/-kSOTtYzgEw/hqdefault.jpg)](https://www.youtube.com/watch?v=-kSOTtYzgEw "CrewAI Getting Started Tutorial")

### Learning Resources

*   [Multi AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
*   [Practical Multi AI Agents and Advanced Use Cases with CrewAI](https://www.deeplearning.ai/short-courses/practical-multi-ai-agents-and-advanced-use-cases-with-crewai/)

### Understanding Flows and Crews

*   **Crews:** Teams of autonomous AI agents collaborating on complex tasks.  Key features include:
    *   Autonomous decision-making
    *   Dynamic task delegation
    *   Specialized roles
    *   Flexible problem-solving

*   **Flows:** Event-driven workflows providing precise control. Key features include:
    *   Fine-grained control over execution paths
    *   Secure state management
    *   Integration with Python code
    *   Conditional branching

The true power of CrewAI emerges when combining Crews and Flows.

### Getting Started with Installation

1.  **Installation**:

    ```shell
    pip install crewai
    ```
    For optional tools:
    ```shell
    pip install 'crewai[tools]'
    ```

### Troubleshooting Dependencies

1.  **ModuleNotFoundError: No module named 'tiktoken'**
    *   Install tiktoken explicitly: `pip install 'crewai[embeddings]'`
    *   If using embedchain or other tools: `pip install 'crewai[tools]'`
2.  **Failed building wheel for tiktoken**
    *   Ensure Rust compiler is installed (see installation steps above)
    *   For Windows: Verify Visual C++ Build Tools are installed
    *   Try upgrading pip: `pip install --upgrade pip`
    *   If issues persist, use a pre-built wheel: `pip install tiktoken --prefer-binary`

### 2. Setting Up Your Crew with the YAML Configuration

```shell
crewai create crew <project_name>
```

This command creates a project folder.  You can customize your project by editing the files in the project folder.  See the original README for the file structure and example.

### 3. Running Your Crew

```bash
cd my_project
crewai install (Optional)
```

```bash
crewai run
```

or

```bash
python src/my_project/main.py
```

## Key Features

CrewAI delivers a standalone, high-performance, multi-AI agent framework, offering simplicity, flexibility, and control.

*   **Standalone & Lean**: Independent of frameworks like LangChain.
*   **Flexible & Precise**: Orchestrate agents via Crews or Flows.
*   **Seamless Integration**: Combine Crews (autonomy) and Flows (precision).
*   **Deep Customization**: Tailor workflows and agent behaviors.
*   **Reliable Performance**: Consistent results.
*   **Thriving Community**: Robust documentation and community support.

## Examples

*   [Landing Page Generator](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/landing_page_generator)
*   [Having Human input on the execution](https://docs.crewai.com/how-to/Human-Input-on-Execution)
*   [Trip Planner](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner)
*   [Stock Analysis](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis)

### Quick Tutorial

[![CrewAI Tutorial](https://img.youtube.com/vi/tnejrr-0a94/maxresdefault.jpg)](https://www.youtube.com/watch?v=tnejrr-0a94 "CrewAI Tutorial")

### Write Job Descriptions

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/job-posting) or watch a video below:

[![Jobs postings](https://img.youtube.com/vi/u98wEMz-9to/maxresdefault.jpg)](https://www.youtube.com/watch?v=u98wEMz-9to "Jobs postings")

### Trip Planner

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/trip_planner) or watch a video below:

[![Trip Planner](https://img.youtube.com/vi/xis7rWp-hjs/maxresdefault.jpg)](https://www.youtube.com/watch?v=xis7rWp-hjs "Trip Planner")

### Stock Analysis

[Check out code for this example](https://github.com/crewAIInc/crewAI-examples/tree/main/crews/stock_analysis) or watch a video below:

[![Stock Analysis](https://img.youtube.com/vi/e0Uj4yWdaAg/maxresdefault.jpg)](https://www.youtube.com/watch?v=e0Uj4yWdaAg "Stock Analysis")

### Using Crews and Flows Together

CrewAI's power comes from combining Crews and Flows.  See the original README for an example.

## Connecting Your Crew to a Model

CrewAI supports connecting to various LLMs.  See the [Connect CrewAI to LLMs](https://docs.crewai.com/how-to/LLM-Connections/) page for details.

## How CrewAI Compares

*   **CrewAI's Advantage:** Combines autonomous agent intelligence with precise workflow control. Excels at orchestration and customization.

*   **LangGraph:** Requires significant boilerplate and complex state management.
*   **Autogen:** Lacks an inherent concept of process, leading to complexity.
*   **ChatDev:** Rigid implementation with limited customization.

## Contribution

CrewAI is open-source and welcomes contributions.  See the original README for contribution guidelines.

```bash
uv lock
uv sync
```

```bash
uv venv
```

```bash
pre-commit install
```

```bash
uv run pytest .
```

```bash
uvx mypy src
```

```bash
uv build
```

```bash
pip install dist/*.tar.gz
```

## Telemetry

CrewAI uses anonymous telemetry data for improvement purposes. Sensitive data such as prompts and tasks are never collected unless explicitly enabled.

## License

CrewAI is released under the [MIT License](https://github.com/crewAIInc/crewAI/blob/main/LICENSE).

## Frequently Asked Questions (FAQ)

See the original README for a detailed FAQ.