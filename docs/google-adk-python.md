# Agent Development Kit (ADK): Build, Evaluate, and Deploy Powerful AI Agents

**ADK empowers developers to build sophisticated AI agents with flexibility and control through a code-first, open-source Python toolkit.**  [View the original repository](https://github.com/google/adk-python).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  <h3>An open-source Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.</h3>
  <h3>
    Important Links:
    <a href="https://google.github.io/adk-docs/">Docs</a>,
    <a href="https://github.com/google/adk-samples">Samples</a>,
    <a href="https://github.com/google/adk-java">Java ADK</a> &
    <a href="https://github.com/google/adk-web">ADK Web</a>.
  </h3>
</div>

ADK is a flexible, modular framework designed for the code-first development of AI agents. Optimized for the Google ecosystem, it is also model-agnostic and deployment-agnostic.  ADK makes agent development accessible, allowing developers to create, deploy, and orchestrate agentic architectures ranging from simple to complex workflows.

## Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs to give agents diverse capabilities.

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for flexibility, testability, and versioning.

*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.

*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to learn how they work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This version is recommended for most users as it represents the most recent official release. The release cadence is weekly.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version to test the newest features, or access critical fixes before they are officially released. Note: the development version may contain experimental changes or bugs not present in the stable release.

## Documentation

Explore the comprehensive documentation for detailed guides:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlights

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

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! See our [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/). If you want to contribute code, please read the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.