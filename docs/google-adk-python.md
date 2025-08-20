# Agent Development Kit (ADK): Build Powerful AI Agents with Ease

**Unleash the power of AI with the Agent Development Kit (ADK), a code-first Python toolkit for creating, testing, and deploying advanced AI agents.** [(Original Repository)](https://github.com/google/adk-python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK provides a flexible, modular, and model-agnostic framework for developers to build AI agents ranging from simple to complex multi-agent systems. Optimized for the Google ecosystem and designed for easy integration with other frameworks.

## Key Features

*   **Code-First Development:** Define agents, tools, and orchestration directly in Python, enabling flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specs, or existing tools to empower your agents with diverse capabilities, with seamless integration within the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent2Agent (A2A) Protocol Integration:** Leverage the A2A protocol for remote agent-to-agent communication. See [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents).
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with an integrated UI.
*   **Evaluation Tools:** Evaluate agent performance.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

Weekly releases ensure you're using the most recent, stable version.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Install the latest features and bug fixes directly from the `main` branch. However, note that this version may contain experimental features or bugs.

## Quickstart Examples

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

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Web UI Example"/>
</div>

## Documentation

Explore comprehensive guides for building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## Contributing

We welcome contributions! Review our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use [llms.txt](./llms.txt) (summarized) or [llms-full.txt](./llms-full.txt) (full info) as context for LLM-powered agent development.

## License

Licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*