<!-- Improved README.md -->
# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash your creativity and build advanced, flexible AI agents with the Agent Development Kit (ADK), a code-first Python toolkit.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK empowers developers to create, deploy, and orchestrate AI agents ranging from simple tasks to complex workflows, offering flexibility and control. Built for the Google ecosystem, ADK is also model-agnostic and deployment-agnostic, designed for seamless integration with other frameworks.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs to give your agents diverse capabilities, with tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## Core Concepts

*   **Agent2Agent (A2A) Protocol Integration:**  ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See the [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.
*   **Development UI:** A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).
    <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

## Getting Started

### Installation

Install the ADK using `pip`:

```bash
pip install google-adk
```

The release cadence is roughly bi-weekly.

Alternatively, install directly from the `main` branch for the latest features and bug fixes:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

### Example: Define a single agent

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

### Example: Define a multi-agent system

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

## Documentation

Comprehensive documentation is available to guide you:

*   **[Documentation](https://google.github.io/adk-docs)**

## Contributing

We welcome contributions!  Please review the:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

For agent development via vibe coding, the following files can be used as context to LLM:

*   [`llms.txt`](./llms.txt): summarized information.
*   [`llms-full.txt`](./llms-full.txt): full information.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*