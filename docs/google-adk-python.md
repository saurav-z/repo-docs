# Agent Development Kit (ADK) - Build and Deploy Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible and code-first toolkit by Google.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<p align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</p>

ADK is an open-source, Python-based toolkit designed to streamline the development, evaluation, and deployment of sophisticated AI agents. Built for flexibility and control, ADK empowers developers to build everything from simple task automation to complex, multi-agent systems. While optimized for the Google ecosystem (Gemini, Vertex AI), ADK is model-agnostic and designed for compatibility with other frameworks.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, providing maximum flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate diverse capabilities using pre-built tools, custom functions, OpenAPI specifications, or existing tools, with tight Google ecosystem integration.
*   **Modular Multi-Agent Systems:** Design and build scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with the built-in development UI.

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This is the recommended approach for most users, providing access to the latest official release with weekly updates.

### Development Version

For access to the latest features and bug fixes before an official PyPI release, install directly from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental changes or bugs. Use it primarily for testing or accessing critical fixes.

## Documentation

Explore the comprehensive documentation for detailed guides on building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Quickstart: Defining Agents

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

### Evaluate Agents:

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! See the following resources for details:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) files as context for LLM-based agent development.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.