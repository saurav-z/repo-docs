# Agent Development Kit (ADK): Build Powerful AI Agents with Ease

**ADK empowers developers to create, evaluate, and deploy sophisticated AI agents with a flexible, code-first approach.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is an open-source, Python-based toolkit designed for building, evaluating, and deploying AI agents. While optimized for Google's ecosystem and Gemini models, ADK is model-agnostic and deployment-agnostic. It allows developers to create agentic architectures ranging from simple tasks to complex workflows with ease.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities, with tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Flexible Deployment:** Containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent-to-Agent Communication:** Seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for advanced agent interaction.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with the built-in development UI.
*   **Evaluation Capabilities:** Evaluate agent performance.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

This release cadence is weekly.

### Development Version

Install directly from the main branch for the latest bug fixes and features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental changes or bugs.

## 📚 Documentation

Explore the full documentation for detailed guides:

*   **[Documentation](https://google.github.io/adk-docs)**

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

Define a multi-agent system with coordinator agent, greeter agent, and task execution agent. Then ADK engine and the model will guide the agents works together to accomplish the task.

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

### Development UI

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions! Please see our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.