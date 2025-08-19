<!-- README.md -->

# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unlock the potential of AI agents with Google's Agent Development Kit (ADK), a flexible, open-source toolkit for building, evaluating, and deploying sophisticated agentic systems.**  Explore the [original repository](https://github.com/google/adk-python) for the latest updates and contributions.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  <a href="https://google.github.io/adk-docs/">Docs</a> |
  <a href="https://github.com/google/adk-samples">Samples</a> |
  <a href="https://github.com/google/adk-java">Java ADK</a> |
  <a href="https://github.com/google/adk-web">ADK Web</a>
</div>

ADK is a versatile, code-first Python toolkit designed for building and deploying cutting-edge AI agents. It's optimized for the Google ecosystem, but remains model-agnostic and deployment-agnostic, providing developers with unparalleled flexibility and control.

## Key Features

*   **Rich Tool Ecosystem:**  Seamlessly integrate pre-built tools, custom functions, OpenAPI specifications, and existing tools to equip your agents with diverse capabilities.
*   **Code-First Development:**  Define agent logic, tools, and orchestration directly in Python for enhanced flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See the [A2A Python Samples](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for example implementations.

## Installation

### Stable Release (Recommended)

Install the latest stable ADK release using `pip`:

```bash
pip install google-adk
```

The release cycle is weekly.

### Development Version

To access the latest features and bug fixes, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may include experimental features or bugs. Use this version for testing and to access critical fixes.

## Documentation

Comprehensive documentation is available for building, evaluating, and deploying agents:

*   [ADK Documentation](https://google.github.io/adk-docs)

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

### Build a Multi-Agent System

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

ADK offers a built-in development UI for testing, evaluation, and debugging your agents:

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI" width="600"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! Refer to the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## Vibe Coding

For developing agents with Vibe coding, use the context provided in [llms.txt](./llms.txt) (summarized) and [llms-full.txt](./llms-full.txt) (full) for your LLM.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*