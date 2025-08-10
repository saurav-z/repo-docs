# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Quickly build, test, and deploy cutting-edge AI agents using the Agent Development Kit (ADK), a flexible and code-first Python toolkit.**  [Learn more at the original repository](https://github.com/google/adk-python).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK empowers developers to create sophisticated AI agents, offering control, flexibility, and ease of deployment.  Designed for optimal performance with Google's AI ecosystem (Gemini, Vertex AI), ADK is model-agnostic and deploy-agnostic, promoting compatibility with other frameworks.

## ✨ Key Features

*   **Code-First Development:** Define agents, tools, and orchestrations directly in Python for enhanced flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specifications for diverse agent capabilities, with a focus on integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible, hierarchical structures.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or seamlessly scale with Vertex AI Agent Engine.
*   **Integrated Development UI**: Easily test, evaluate, debug, and showcase your agent(s).

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  Explore an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how they work together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable ADK version using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

Access the latest features and bug fixes from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may contain experimental features or bugs. Use it primarily for testing or accessing critical fixes.*

## 📚 Documentation

Comprehensive guides are available:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Quickstart Examples

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

## 🤝 Contributing

Contribute to the ADK community! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) to contribute code.

## Vibe Coding

For agent development using vibe coding, leverage the provided context files:

*   [llms.txt](./llms.txt) (summarized information)
*   [llms-full.txt](./llms-full.txt) (full information)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*