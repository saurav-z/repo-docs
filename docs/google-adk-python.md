# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Empower your AI development with the Agent Development Kit (ADK), a code-first Python toolkit for creating sophisticated, flexible, and scalable AI agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

<div align="center">
  <a href="https://google.github.io/adk-docs/">Docs</a> |
  <a href="https://github.com/google/adk-samples">Samples</a> |
  <a href="https://github.com/google/adk-java">Java ADK</a> |
  <a href="https://github.com/google/adk-web">ADK Web</a>
</div>

ADK is a flexible and modular framework designed to streamline the development and deployment of AI agents.  It’s optimized for the Google ecosystem, yet is model-agnostic and deployment-agnostic, providing compatibility with other frameworks.  Designed to make agent development feel more like software development, ADK empowers developers to build, deploy, and orchestrate agentic architectures, from simple tasks to complex workflows.

## Key Features:

*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, OpenAPI specs, and integrate existing tools to equip agents with diverse capabilities, with tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with a built-in development UI.
*   **Agent Evaluation:** Evaluate the performance of your agents.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See the [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how it works.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

This is the recommended version for most users, offering the latest official release.  The release cadence is weekly.

### Development Version

Install directly from the `main` branch to access the latest bug fixes and features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may include experimental changes or bugs.*

## 📚 Documentation

Explore the full documentation for detailed guides:

*   [Documentation](https://google.github.io/adk-docs)

## 🏁 Feature Highlight

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

We welcome community contributions!

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*