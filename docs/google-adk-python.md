# Agent Development Kit (ADK): Build Sophisticated AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a code-first Python toolkit for building, evaluating, and deploying cutting-edge AI solutions.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is an open-source, flexible, and modular framework designed for developers to create and deploy AI agents, supporting a wide range of applications from simple tasks to complex workflows.  It's optimized for the Google ecosystem, while remaining model-agnostic and deployment-agnostic.

**Key Features:**

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, promoting flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities, with strong integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with a user-friendly UI.
*   **Agent2Agent (A2A) Protocol Integration:** Communicate and collaborate with other agents using the A2A protocol.

## 🤖 Agent2Agent (A2A) Protocol

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See the [A2A Python sample](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how to get started.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

The release cadence is weekly. This version is recommended for most users as it represents the most recent official release.

### Development Version

Install directly from the main branch to access bug fixes and new features before they are officially released:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental changes or bugs. Use it primarily for testing or accessing critical fixes.

## 📚 Documentation & Resources

*   **[Documentation](https://google.github.io/adk-docs)**: Comprehensive guides for building, evaluating, and deploying agents.
*   [Samples](https://github.com/google/adk-samples)
*   [Java ADK](https://github.com/google/adk-java)
*   [ADK Web](https://github.com/google/adk-web)

## 🏁 Example: Build and Evaluate an Agent

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

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  See:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.