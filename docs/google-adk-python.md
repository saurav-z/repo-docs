<!--
  SPDX-License-Identifier: Apache-2.0
-->
# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Empower your AI development with the Agent Development Kit (ADK), a versatile Python toolkit for building, evaluating, and deploying sophisticated AI agents.**

[<img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License">](LICENSE)
[<img src="https://img.shields.io/pypi/v/google-adk" alt="PyPI">](https://pypi.org/project/google-adk/)
[<img src="https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg" alt="Python Unit Tests">](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[<img src="https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white" alt="Reddit">](https://www.reddit.com/r/agentdevelopmentkit/)
[<img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK is an open-source, code-first Python toolkit designed for building, evaluating, and deploying sophisticated AI agents. Optimized for Gemini and the Google ecosystem, ADK offers flexibility, control, and seamless integration, allowing developers to create, deploy, and orchestrate agentic architectures ranging from simple tasks to complex workflows.

**Explore ADK:**
*   [Documentation](https://google.github.io/adk-docs/)
*   [Samples](https://github.com/google/adk-samples)
*   [Java ADK](https://github.com/google/adk-java)
*   [ADK Web](https://github.com/google/adk-web)

## Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications to equip agents with diverse capabilities, all within the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible, interconnected hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent-to-Agent (A2A) Protocol Integration:** Facilitates remote agent communication using the A2A protocol for advanced interactions.

## Getting Started

### Installation

#### Stable Release (Recommended)

Install the latest stable version via `pip`:

```bash
pip install google-adk
```

This is the recommended approach for most users, providing access to the latest official releases.  The release cadence is weekly.

#### Development Version

To access the latest features and bug fixes, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:**  The development version may include experimental features and is suitable for testing and accessing the newest updates.

## Code Examples

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

### Development UI

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

ADK includes a built-in development UI to test, evaluate, debug, and showcase your agents.

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Agent2Agent (A2A) Protocol

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) to facilitate remote agent-to-agent communication. See the [A2A Example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for integration details.

## Contributing

We welcome contributions!  Review the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## Vibe Coding

For vibe coding, use [llms.txt](./llms.txt) or [llms-full.txt](./llms-full.txt) as context for your LLM.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

**[Back to the project](https://github.com/google/adk-python)**

*Happy Agent Building!*