# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible, open-source toolkit for Python that empowers developers to build, evaluate, and deploy sophisticated agentic architectures.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<p align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</p>

## Key Features

*   **Code-First Development:** Build agents and orchestrate complex workflows directly in Python, offering ultimate flexibility, version control, and testability.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specs, or existing tools to equip your agents with diverse capabilities, especially optimized for the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies, enabling complex task execution.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent2Agent (A2A) Protocol Integration:** Leverage the A2A protocol for seamless communication between agents, enabling advanced collaboration.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with a convenient development UI.
*   **Evaluation Tools:** Built-in evaluation capabilities for testing your agent's performance.

## Getting Started

### Installation

#### Stable Release (Recommended)

```bash
pip install google-adk
```

#### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## Examples

### Defining a Single Agent

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

### Defining a Multi-Agent System

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

### Evaluating Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Documentation

*   [ADK Documentation](https://google.github.io/adk-docs)

## Contributing

We welcome contributions!  See our:

*   [General Contribution Guidelines](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding Context

For vibe coding, the following files provide context:

*   [llms.txt](./llms.txt) (Summarized LLM information)
*   [llms-full.txt](./llms-full.txt) (Full LLM information)

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.