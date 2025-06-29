# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**[Explore the Agent Development Kit (ADK) on GitHub](https://github.com/google/adk-python)** – a code-first Python toolkit empowering developers to build, evaluate, and deploy sophisticated AI agents.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK provides a flexible and modular framework, enabling developers to create and deploy AI agents ranging from simple task automation to complex, multi-agent workflows. While optimized for the Google ecosystem, ADK is built to be model-agnostic and deployment-agnostic.

## Key Features of the Agent Development Kit (ADK)

*   **Code-First Agent Development:** Define agents, their logic, tools, and orchestration directly in Python for enhanced flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications, or leverage existing tools to give agents diverse capabilities, with seamless integration into the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies for advanced capabilities.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for secure remote agent-to-agent communication.  See this [example](https://github.com/google-a2a/a2a-samples/tree/main/samples/python/agents/google_adk) to see how they can work together.

## 🚀 Getting Started: Installation

### Stable Release (Recommended)

Install the latest stable version using pip:

```bash
pip install google-adk
```

This version is recommended for most users, as it represents the most recent official release.

### Development Version

To access the latest features and bug fixes, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may include experimental changes or bugs. Use it primarily for testing upcoming changes or accessing critical fixes before the official release.*

## 📚 Documentation and Resources

*   **[Documentation](https://google.github.io/adk-docs):** Comprehensive guides on building, evaluating, and deploying agents.
*   **[Samples](https://github.com/google/adk-samples):** Explore example implementations of ADK.
*   **[Java ADK](https://github.com/google/adk-java):**  The ADK is available in Java as well.
*   **[ADK Web](https://github.com/google/adk-web):**  Web-based interface for the ADK.

## 🏁 Feature Highlights

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

### Multi-Agent Systems

Define and coordinate multiple specialized agents within a system.

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

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  Please review our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*