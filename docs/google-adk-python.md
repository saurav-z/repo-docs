# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**ADK empowers developers to create, evaluate, and deploy sophisticated AI agents with a code-first approach.**  Learn more and contribute on the [original repository](https://github.com/google/adk-python).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK is an open-source, code-first Python toolkit designed for building and deploying AI agents.  It's flexible, modular, and designed for tight integration with the Google ecosystem while remaining model-agnostic and deployment-agnostic, allowing for use with other frameworks. ADK makes agent development feel more like software development, simplifying the creation, deployment, and orchestration of agentic architectures.

**Key Features:**

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python.
*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, and OpenAPI specs.
*   **Modular Multi-Agent Systems:** Compose specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents.
*   **Agent2Agent (A2A) Protocol Integration:** Supports remote agent-to-agent communication via the [A2A protocol](https://github.com/google-a2a/A2A/).

## 🚀 Getting Started

### Installation

#### Stable Release (Recommended)

```bash
pip install google-adk
```

This is the recommended method for most users, providing the most recent official release. Releases are made weekly.

#### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Install the latest changes from the `main` branch for bug fixes and new features not yet in a stable release.  Note that this version may be less stable.

## 📚 Documentation

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

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

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions! See the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.