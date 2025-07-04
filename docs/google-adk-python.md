# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Build, evaluate, and deploy sophisticated AI agents with the Google Agent Development Kit (ADK), a flexible, code-first toolkit.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

[<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>](https://github.com/google/adk-python)

ADK is an open-source, Python-based toolkit designed to simplify the development, evaluation, and deployment of AI agents.  It is optimized for the Google ecosystem but is model-agnostic and deployment-agnostic, offering developers flexibility and control in building complex agentic applications.

**Key Features:**

*   **Rich Tool Ecosystem:**  Integrate pre-built tools, custom functions, and OpenAPI specs to give agents diverse capabilities, with strong integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Get Started

### Installation

Choose the installation method that suits your needs:

#### Stable Release (Recommended)

Install the latest official release via `pip`:

```bash
pip install google-adk
```

#### Development Version

Install the latest changes directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Important:** The development version may contain experimental features or bugs. Use it primarily for testing or accessing critical fixes before they are officially released.

### Quickstart Examples

**Define a Single Agent:**

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

**Define a Multi-Agent System:**

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

ADK includes a built-in development UI to help you test, evaluate, debug, and showcase your agent(s):

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI Screenshot"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to learn how to integrate A2A and ADK.

## Documentation

Explore the full documentation:

*   **[ADK Documentation](https://google.github.io/adk-docs/)**

## Contributing

We welcome contributions!  Review the [contribution guidelines](https://google.github.io/adk-docs/contributing-guide/) and [code contributing guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*

**[Back to the Top](https://github.com/google/adk-python)**