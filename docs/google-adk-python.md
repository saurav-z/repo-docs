# Agent Development Kit (ADK): Build, Evaluate, and Deploy Powerful AI Agents

**[Agent Development Kit (ADK) on GitHub](https://github.com/google/adk-python)** - Unleash your AI potential with the ADK, a code-first, open-source Python toolkit for crafting cutting-edge AI agents.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

The Agent Development Kit (ADK) empowers developers to create, evaluate, and deploy sophisticated AI agents with unparalleled flexibility and control, making agent development feel like traditional software development.

## ✨ Key Features of ADK

*   **Code-First Development:** Define agents, tools, and orchestration logic directly in Python for maximum flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs to give agents diverse capabilities, with tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Configuration:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) using the built-in development UI.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of them working together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

The release cadence is weekly. This version is recommended for most users.

### Development Version

Install directly from the main branch for the newest fixes and features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** Use the development version for testing or critical fixes before official releases.

## 📚 Documentation

Explore the comprehensive [documentation](https://google.github.io/adk-docs) for detailed guides on building, evaluating, and deploying agents.

## 🏁 Quickstart Examples

### Define a Single Agent

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

### Define a Multi-Agent System

Define a multi-agent system with coordinator, greeter, and task execution agents. ADK orchestrates their collaboration:

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

### Agent Evaluation

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use the [llms.txt](./llms.txt) or [llms-full.txt](./llms-full.txt) files as context when developing agents with LLMs.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*