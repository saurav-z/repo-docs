# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**[Explore the ADK on GitHub](https://github.com/google/adk-python) and unlock the power of agent-based AI with Google's open-source toolkit.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

The Agent Development Kit (ADK) is a comprehensive Python toolkit that empowers developers to build, evaluate, and deploy sophisticated AI agents with unparalleled flexibility and control. Designed for rapid development and seamless integration with the Google ecosystem, ADK enables the creation of both simple and complex agentic architectures.

## Key Features of the Agent Development Kit (ADK)

*   **Code-First Agent Development:** Define agent logic, tools, and orchestration directly in Python for enhanced control, testability, and versioning.
*   **Rich Tool Integration:** Utilize pre-built tools, integrate custom functions, leverage OpenAPI specifications, or connect with existing tools, providing diverse capabilities for your agents.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies, enabling complex workflows and efficient task management.
*   **Flexible Deployment Options:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Model-Agnostic Design:** Build agents that are compatible with various LLM models, allowing for flexibility and future-proofing of your agent applications.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication, enabling advanced distributed agent architectures. Explore the [A2A Example](https://github.com/google-a2a/a2a-samples/tree/main/samples/python/agents/google_adk) for how to work with ADK and A2A.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable release of ADK using `pip`:

```bash
pip install google-adk
```

*   Weekly release cadence.

### Development Version

Install the latest development version from the main branch for bug fixes and new features (use with caution):

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*   The development version may include experimental features or bugs.

## 📚 Documentation

Access detailed guides and tutorials on building, evaluating, and deploying agents:

*   **[ADK Documentation](https://google.github.io/adk-docs/)**

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

ADK includes a development UI for testing, evaluation, debugging, and showcasing your agents:

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions! Learn how to contribute by reading the:
*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.