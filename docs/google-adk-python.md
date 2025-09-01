<!-- Agent Development Kit (ADK) - Build, Evaluate, & Deploy AI Agents -->

# Agent Development Kit (ADK) - Open-Source AI Agent Development

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible, code-first toolkit for building cutting-edge AI applications.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  <a href="https://google.github.io/adk-docs/">Docs</a> |
  <a href="https://github.com/google/adk-samples">Samples</a> |
  <a href="https://github.com/google/adk-java">Java ADK</a> |
  <a href="https://github.com/google/adk-web">ADK Web</a>
</div>

The Agent Development Kit (ADK) provides a modular and flexible framework for developing and deploying AI agents. ADK is designed to make agent development feel more like software development, simplifying the creation, deployment, and orchestration of agentic architectures. While optimized for the Google ecosystem and Gemini models, ADK is built to be model-agnostic and deployment-agnostic.

**[Explore the ADK on GitHub](https://github.com/google/adk-python)**

## ✨ Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Integrate a diverse range of tools, including pre-built options, custom functions, and OpenAPI specs, to give agents diverse capabilities.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code using the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with a built-in development UI.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using pip:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

Install directly from the `main` branch for bug fixes and new features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version for testing and accessing the newest changes, but be aware of potential instability.

## 📚 Documentation

Comprehensive documentation is available for detailed guides on building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

### Define a Single Agent:

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

### Define a Multi-Agent System:

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

## Vibe Coding

Use the `llms.txt` and `llms-full.txt` files as context for LLMs when developing agents via vibe coding.  The former is summarized, and the latter provides full information.

## 🤝 Contributing

We welcome contributions from the community!  See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) to get started contributing code.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.