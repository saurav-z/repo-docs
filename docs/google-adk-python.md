# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible, code-first Python toolkit from Google.  [Learn More](https://github.com/google/adk-python)**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is a powerful, open-source toolkit designed for building, evaluating, and deploying sophisticated AI agents.  Optimized for the Google ecosystem, ADK is also model-agnostic and deployment-agnostic, promoting flexibility and control for developers.  It enables a code-first approach, making agent development more like traditional software development, simplifying the creation, deployment, and orchestration of agents for tasks ranging from simple to complex.

## Key Features

*   ✅ **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, maximizing flexibility, testability, and versioning.
*   ✅ **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs, seamlessly connecting agents to diverse capabilities and the Google ecosystem.
*   ✅ **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   ✅ **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale with Vertex AI Agent Engine.
*   ✅ **Built-in Development UI**: Test, evaluate, debug, and showcase your agent(s).
*   ✅ **Agent Evaluation**: Evaluate agents.
*   ✅ **Agent-to-Agent Communication:** Supports the A2A protocol for remote agent interaction.

## Getting Started

### Installation

#### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

This version is recommended for general use.

#### Development Version

Install directly from the `main` branch for the latest updates:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may include experimental features or bugs.*

## Example: Define a single agent

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

## Example: Define a multi-agent system

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

## Example: Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>
</div>

## 📚 Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying your agents:

*   [Documentation](https://google.github.io/adk-docs)

## 🤝 Contributing

We welcome contributions!  Please see our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*