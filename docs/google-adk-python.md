# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible and code-first Python toolkit for building sophisticated AI applications.**  ([Original Repo](https://github.com/google/adk-python))

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
  <b>Important Links:</b>
  <a href="https://google.github.io/adk-docs/">Docs</a>, 
  <a href="https://github.com/google/adk-samples">Samples</a>,
  <a href="https://github.com/google/adk-java">Java ADK</a> &
  <a href="https://github.com/google/adk-web">ADK Web</a>.
</div>

The Agent Development Kit (ADK) is a powerful and versatile framework designed for developers seeking to create, deploy, and orchestrate AI agents. While ADK is optimized for the Google ecosystem, it's built to be model-agnostic and deployment-agnostic, ensuring compatibility with a wide range of tools and platforms.  ADK empowers developers to build agentic architectures that range from simple tasks to complex workflows.

## Key Features

*   **Rich Tool Ecosystem:** Integrate diverse capabilities using pre-built tools, custom functions, OpenAPI specifications, and existing tools, all optimized for tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for unparalleled flexibility, testability, and versioning control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies, enabling complex workflows.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. Explore this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they work together.

## Installation

### Stable Release (Recommended)

Install the latest stable ADK release via pip:

```bash
pip install google-adk
```

The release cadence is weekly.

This is the recommended installation method for most users.

### Development Version

Access the latest bug fixes and features by installing directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:**  The development version may include experimental features or bugs. Use it for testing and accessing the newest updates.

## Documentation

Comprehensive guides on building, evaluating, and deploying agents:

*   [Documentation](https://google.github.io/adk-docs)

## Feature Highlight

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

### Development UI

Utilize a built-in development UI for testing, evaluation, debugging, and showcasing your agents.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI" />

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions! Refer to our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) to contribute code.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*