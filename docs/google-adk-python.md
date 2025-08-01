# Agent Development Kit (ADK): Build Powerful AI Agents with Ease

**Empower your projects with the Agent Development Kit (ADK), a versatile Python toolkit for creating, evaluating, and deploying cutting-edge AI agents.**  [View the original repo](https://github.com/google/adk-python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  <h3>An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.</h3>
  <h3>
    Important Links:
    <a href="https://google.github.io/adk-docs/">Docs</a>,
    <a href="https://github.com/google/adk-samples">Samples</a>,
    <a href="https://github.com/google/adk-java">Java ADK</a> &
    <a href="https://github.com/google/adk-web">ADK Web</a>.
  </h3>
</div>

ADK is a flexible and modular framework designed for developing and deploying AI agents. Optimized for the Gemini and Google ecosystem, ADK is also model-agnostic and deployment-agnostic, ensuring compatibility with various frameworks. It empowers developers to create, deploy, and orchestrate agentic architectures, from simple tasks to complex workflows.

---

## ✨ Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs, or leverage existing tools for diverse agent capabilities, with tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, enabling ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI**: Test, evaluate, debug, and showcase your agent(s) with the development UI

## 🤖 Agent-to-Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. Explore how agents can work together in this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents).

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly. This version is recommended for most users as it represents the most recent official release.

### Development Version

Access bug fixes and new features merged into the `main` branch by installing directly from GitHub:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental features or bugs. Use it primarily for testing or accessing critical fixes before official releases.

## 📚 Documentation

Comprehensive documentation is available for building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Quickstart: Examples

### Define a Single Agent:

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

### Define a Multi-Agent System:

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
### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  See our:
- [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
- [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

For agent development using vibe coding, use the following context files for your LLM:
*   [llms.txt](./llms.txt) (summarized)
*   [llms-full.txt](./llms-full.txt) (full info)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*