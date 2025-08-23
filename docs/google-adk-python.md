# Agent Development Kit (ADK): Build, Evaluate, and Deploy Powerful AI Agents

**Unlock the power of AI agents with the Agent Development Kit (ADK), a flexible, code-first Python toolkit for building sophisticated and production-ready agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK empowers developers to create, deploy, and orchestrate AI agents, ranging from simple task automation to complex multi-agent systems. Designed with flexibility and control in mind, ADK offers a seamless development experience, optimized for Gemini and the Google ecosystem, but adaptable to other frameworks.

## Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI integrations to equip agents with diverse capabilities, including tight Google ecosystem integration.
*   **Modular Multi-Agent Systems:** Design scalable and sophisticated applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config**: Build agents without code using the Agent Config feature.

## ✨ What's New

*   **Agent Config**: Build agents without code. Check out the
    [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to learn how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using pip:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

For access to the latest features and bug fixes, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version includes the newest changes but may contain experimental features or bugs.

## 📚 Documentation

Explore comprehensive documentation for detailed guides on building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlight

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

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome community contributions! Refer to our:
-   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
-   [Code Contributing Guidelines](./CONTRIBUTING.md) to start contributing code.

## Vibe Coding

If you are to develop agent via vibe coding the [llms.txt](./llms.txt) and the [llms-full.txt](./llms-full.txt) can be used as context to LLM. While the former one is a summarized one and the later one has the full information in case your LLM has big enough context window.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*