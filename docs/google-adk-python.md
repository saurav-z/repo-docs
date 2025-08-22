# Agent Development Kit (ADK): Build and Deploy AI Agents with Ease

**Empower your AI development with the Agent Development Kit (ADK), a flexible, code-first toolkit for building sophisticated AI agents.**  [See the original repository on GitHub](https://github.com/google/adk-python).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is an open-source, Python-based toolkit designed for developing, evaluating, and deploying advanced AI agents. While optimized for Google's ecosystem (especially Gemini), ADK is model-agnostic and designed to integrate seamlessly with other frameworks, making agent development more akin to traditional software development. Create and orchestrate agents ranging from simple tasks to complex workflows.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, offering maximum flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities, including tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Build scalable applications by composing multiple specialized agents into flexible hierarchies for complex task management.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents using the development UI.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See the [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to explore the integration.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

Install directly from the `main` branch for bug fixes and new features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental changes or bugs. Use it primarily for testing upcoming changes or accessing critical fixes.

## 📚 Documentation

Comprehensive documentation is available for detailed guides:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Quick Start

### Define a single agent:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a multi-agent system:

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="greeter", model="gemini-2.0-flash", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.0-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions.  Refer to the:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) for code contributions.

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for your LLM during agent development.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*