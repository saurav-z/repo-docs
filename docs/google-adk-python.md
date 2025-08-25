# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

[Google's Agent Development Kit (ADK)](https://github.com/google/adk-python) empowers developers to create, evaluate, and deploy sophisticated AI agents with a code-first approach, offering flexibility and control.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

## Key Features

*   **Code-First Agent Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specs, or existing tools to give agents diverse capabilities, with tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with a built-in development UI.
*   **Agent Config:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## What's New

*   **Agent Config:** Build agents without code. Check out the
    [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See the [A2A example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to learn how to use them together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## Documentation

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlights

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

Define a multi-agent system with coordinator agent, greeter agent, and task execution agent. Then ADK engine and the model will guide the agents works together to accomplish the task.

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

We welcome contributions! See:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

If you are to develop agent via vibe coding the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) can be used as context to LLM. While the former one is a summarized one and the later one has the full information in case your LLM has big enough context window.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.