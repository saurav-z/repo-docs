# Agent Development Kit (ADK): Build & Deploy Powerful AI Agents

**Unleash the power of AI agents with the Agent Development Kit (ADK), a code-first Python toolkit for building, evaluating, and deploying sophisticated agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK is an open-source, flexible, and modular framework designed to simplify the development, evaluation, and deployment of AI agents. While optimized for the Google ecosystem, ADK is model-agnostic and deployment-agnostic, fostering compatibility with various frameworks and tools.  ADK empowers developers to build and orchestrate agentic architectures, from simple tasks to complex workflows, with ease.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, OpenAPI specifications, and integrate with existing tools to give agents diverse capabilities, with seamless Google ecosystem integration.
*   **Modular Multi-Agent Systems:** Design and build scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI**: Test, evaluate, debug, and showcase your agent(s) with a built-in UI.
*   **Agent2Agent (A2A) Protocol Integration**: Facilitates remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for details.

## Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

This version is recommended for most users and is updated weekly.

### Development Version

Install the latest changes directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version for testing upcoming changes or accessing critical fixes.

## Getting Started: Example Code

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

### Development UI

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

## Documentation

Comprehensive documentation is available for building, evaluating, and deploying agents:

*   [**Documentation**](https://google.github.io/adk-docs)

## Vibe Coding

For agent development via vibe coding, use the provided context files:

*   [`llms.txt`](./llms.txt) (Summarized LLM context)
*   [`llms-full.txt`](./llms-full.txt) (Full LLM context)

## Contributing

We welcome contributions from the community!  See:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.