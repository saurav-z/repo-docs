# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash your creativity and build, evaluate, and deploy sophisticated AI agents with Google's Agent Development Kit (ADK)!** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

<div align="center">
  <p>An open-source, code-first Python toolkit for building, evaluating, and deploying AI agents.</p>
  <p>
    <a href="https://google.github.io/adk-docs/">Docs</a> |
    <a href="https://github.com/google/adk-samples">Samples</a> |
    <a href="https://github.com/google/adk-java">Java ADK</a> |
    <a href="https://github.com/google/adk-web">ADK Web</a>
  </p>
</div>

The Agent Development Kit (ADK) empowers developers to build and deploy AI agents with unparalleled flexibility and control. Designed with a code-first approach, ADK simplifies the development of complex agentic architectures, from simple task automation to sophisticated multi-agent systems. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic and deployment-agnostic, ensuring compatibility with a wide range of frameworks.

## Key Features:

*   **Code-First Development:** Define agents, tools, and orchestration logic directly in Python, fostering flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs, or easily incorporate existing tools to equip agents with diverse capabilities.
*   **Modular Multi-Agent Systems:** Build scalable applications by composing specialized agents into flexible hierarchies and workflows.
*   **Deploy Anywhere:** Containerize and deploy agents seamlessly on Cloud Run or scale with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. Explore how A2A and ADK work together in this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents).

## Installation

### Stable Release (Recommended)

Install the latest stable version using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

Access the latest features and bug fixes by installing directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```
**Note:** The development version may contain experimental features or bugs. Use it for testing or accessing critical fixes.

## Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## Feature Highlights

### Define a Single Agent:

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

### Define a Multi-Agent System:

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

A built-in development UI facilitates testing, evaluation, and debugging of your agents:

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome community contributions!  Review our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md).

## Vibe Coding

Use [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) as context for LLMs during vibe coding.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*