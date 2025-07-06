# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unlock the power of AI agents with the Agent Development Kit (ADK), a code-first Python toolkit for building, evaluating, and deploying sophisticated agentic systems.**  For more details, explore the [original repository](https://github.com/google/adk-python).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</div>

<div align="center">
  <h3>An open-source Python toolkit for building and deploying AI agents.</h3>
  <p>
    <a href="https://google.github.io/adk-docs/">Docs</a> |
    <a href="https://github.com/google/adk-samples">Samples</a> |
    <a href="https://github.com/google/adk-java">Java ADK</a> |
    <a href="https://github.com/google/adk-web">ADK Web</a>
  </p>
</div>

ADK is a flexible and modular framework designed for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic and deployment-agnostic, built for compatibility with other frameworks. It streamlines agent development, making it easier to create, deploy, and orchestrate agentic architectures.

## Key Features

*   ✅ **Code-First Development:** Define agent logic, tools, and orchestration directly in Python.
*   ✅ **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, and integrate existing tools for diverse agent capabilities.
*   ✅ **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   ✅ **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This is the recommended option for most users, providing the latest official release with weekly updates.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Install directly from the `main` branch for the latest bug fixes and features. Note that this version may include experimental changes or bugs.

## Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   [ADK Documentation](https://google.github.io/adk-docs)

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

### Multi-Agent System Example

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! Review our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*