# Agent Development Kit (ADK): Build Powerful AI Agents with Code-First Control

ADK empowers developers to create, evaluate, and deploy sophisticated AI agents with unparalleled flexibility. ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

<div align="center">
  An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents.
</div>

<div align="center">
  Important Links:
  <a href="https://google.github.io/adk-docs/">Docs</a>,
  <a href="https://github.com/google/adk-samples">Samples</a>,
  <a href="https://github.com/google/adk-java">Java ADK</a> &
  <a href="https://github.com/google/adk-web">ADK Web</a>.
</div>

Agent Development Kit (ADK) is a flexible and modular framework designed for developing and deploying AI agents.  ADK is model-agnostic, deployment-agnostic, and built for compatibility with other frameworks.  It empowers developers to create, deploy, and orchestrate agentic architectures, from simple tasks to complex workflows.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, ensuring ultimate flexibility, testability, and version control.
*   **Rich Tool Ecosystem:**  Leverage pre-built tools, custom functions, and OpenAPI specs to give agents diverse capabilities, with seamless integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:**  Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/google-a2a/a2a-samples/tree/main/samples/python/agents/google_adk) for how to utilize them.

## Installation

### Stable Release (Recommended)

Install the latest stable release using pip:

```bash
pip install google-adk
```

This version is recommended for most users and is updated weekly.

### Development Version

For the latest features and bug fixes, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:**  The development version may contain experimental features or bugs.

## Documentation

Comprehensive documentation, including guides for building, evaluating, and deploying agents, is available at:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlights

### Single Agent Definition

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

### Multi-Agent System Definition

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI Example"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! Review the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*