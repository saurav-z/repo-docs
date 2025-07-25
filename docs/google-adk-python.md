# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash your creativity and build sophisticated AI agents with the Agent Development Kit (ADK), a flexible, code-first Python toolkit from Google.**  Learn more and contribute at the [original repository](https://github.com/google/adk-python).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
</div>

<div align="center">
  **Important Links:**
  <a href="https://google.github.io/adk-docs/">Docs</a> |
  <a href="https://github.com/google/adk-samples">Samples</a> |
  <a href="https://github.com/google/adk-java">Java ADK</a> |
  <a href="https://github.com/google/adk-web">ADK Web</a>
</div>

The Agent Development Kit (ADK) provides a modular and flexible framework for developers to create and deploy AI agents.  Designed for seamless integration with the Google ecosystem (Gemini), ADK is also model-agnostic, deployment-agnostic and compatible with other frameworks, allowing you to build agentic architectures from simple tasks to complex workflows.

## Key Features

*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specifications for diverse agent capabilities, tightly integrated with Google services.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with a user-friendly development interface.

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they work together.

## Installation

### Stable Release (Recommended)

Install the latest stable ADK version using `pip`:

```bash
pip install google-adk
```

(Release cadence is weekly)

### Development Version

To access the latest bug fixes and features before official releases, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note:* The development version may contain experimental features or bugs.

## Documentation

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Example: Define and Use an Agent

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

## Contributing

We welcome contributions! Review the following guides:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)