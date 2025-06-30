<!-- SEO-optimized README for Agent Development Kit (ADK) -->

# Agent Development Kit (ADK): Build Powerful AI Agents

**[Agent Development Kit (ADK)](https://github.com/google/adk-python) empowers developers to build, evaluate, and deploy sophisticated AI agents with flexibility and control, using a code-first approach.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK is an open-source Python toolkit designed for building, evaluating, and deploying AI agents. It provides a flexible and modular framework, making it easy to create, deploy, and orchestrate AI agents for a range of tasks, from simple automation to complex workflows. While optimized for the Google ecosystem and Gemini models, ADK is model-agnostic and designed for compatibility with other frameworks.

**Key Benefits:**

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specifications, or existing tools, providing diverse capabilities for agents.
*   **Modular Multi-Agent Systems:** Build scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **A2A Protocol Integration:**  Seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) using the built-in development UI.
*   **Agent Evaluation Tools:** Evaluate agents using a command-line interface (CLI) to measure performance with user-defined test sets.

## Key Features

*   **Flexible Agent Definition:** Easily define single agents or build complex multi-agent systems.
*   **Model Agnostic:** Works with various LLMs (Large Language Models).
*   **Google Ecosystem Integration:** Optimized for Gemini and the Google ecosystem, including tools like Google Search.
*   **Easy Deployment:** Containerize and deploy agents on Cloud Run or Vertex AI Agent Engine.
*   **Agent-to-Agent Communication:** Built-in support for the A2A protocol.
*   **Evaluation Tools:** Includes tools for evaluating agent performance.

## Getting Started

### Installation

**Stable Release (Recommended)**
Install the latest stable version using pip:

```bash
pip install google-adk
```

**Development Version**
Install the latest changes from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```
*Note:* The development version may contain experimental changes or bugs. Use it primarily for testing upcoming changes or accessing critical fixes before official releases.

## Code Examples

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
###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```
##  Development UI

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

## Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## Contributing

We welcome contributions!  Refer to:
-   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
-   [Code Contributing Guidelines](./CONTRIBUTING.md)

## License

This project is licensed under the Apache 2.0 License.  See the [LICENSE](LICENSE) file for details.

---

*Build amazing AI agents with ADK!*