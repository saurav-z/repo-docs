# Agent Development Kit (ADK): Build Powerful AI Agents with Code

**Unleash your creativity and streamline agent development with the Agent Development Kit (ADK), a code-first Python toolkit from Google for building, evaluating, and deploying sophisticated AI agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK is a flexible and modular framework that makes agent development feel like software development. This allows developers to create, deploy, and orchestrate agentic architectures, from simple tasks to complex workflows. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic and designed for compatibility with other frameworks.

## 🔑 Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities and seamlessly connect with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, version control, and rigorous testing.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies for complex workflows.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent-to-Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication, enabling sophisticated interactions.  See an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how they can be used together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly. This version is recommended for most users as it represents the most recent official release.

### Development Version

For the latest features and bug fixes, install directly from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Important:** The development version may contain experimental changes or bugs. Use it for testing or accessing critical fixes before they are officially released.

## 📚 Documentation

Comprehensive documentation is available for building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 💡 Feature Highlights

### Define a Single Agent

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash",  # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a Multi-Agent System

Create sophisticated applications by defining a system with a coordinator agent, greeter agent, and task execution agent.  ADK will orchestrate the agents to accomplish complex tasks.

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
    sub_agents=[  # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

Test, evaluate, debug, and showcase your agents with the built-in development UI.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="Development UI Screenshot">

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  Find guidance in these resources:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 🧠 Vibe Coding

Utilize `llms.txt` and `llms-full.txt` for context when developing agents via vibe coding.  `llms.txt` is a summarized version, while `llms-full.txt` provides complete information.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*