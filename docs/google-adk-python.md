# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible, open-source Python toolkit from Google for building, evaluating, and deploying sophisticated AI agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  <i>An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.</i>
</div>

<div align="center">
  <br/>
  <b>Important Links:</b>
  <a href="https://google.github.io/adk-docs/">Docs</a> |
  <a href="https://github.com/google/adk-samples">Samples</a> |
  <a href="https://github.com/google/adk-java">Java ADK</a> |
  <a href="https://github.com/google/adk-web">ADK Web</a>
  <br/>
</div>

The Agent Development Kit (ADK) is designed to simplify the development lifecycle of AI agents, making it feel more like standard software development.  It is a flexible, modular framework for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and compatible with other frameworks.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specs, or existing tools to empower agents with diverse capabilities, with seamless integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

*   The release cadence is weekly.
*   Recommended for most users for the most recent official release.

### Development Version

Install the development version directly from the `main` branch for bug fixes and new features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*   Includes the newest fixes and features.
*   Use for testing upcoming changes or accessing critical fixes before official releases.  May contain experimental changes or bugs.

## Documentation

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlight

### Define a Single Agent:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash",  # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a Multi-Agent System:

Create a multi-agent system with coordinator, greeter, and task execution agents to accomplish tasks collaboratively.

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
    sub_agents=[  # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

A built-in development UI helps you test, evaluate, debug, and showcase your agents.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI" />

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) to contribute code.

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for LLMs during agent development via vibe coding.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*