# Agent Development Kit (ADK): Build Powerful AI Agents with Ease

**ADK is a code-first Python toolkit enabling developers to create, evaluate, and deploy sophisticated AI agents with flexibility and control, offering tight integration with the Google ecosystem and beyond.** ([View the original repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

ADK empowers developers to build and deploy advanced AI agents.  It is model-agnostic and deployment-agnostic, with a focus on a code-first approach to agent development, making it easier to build, deploy, and orchestrate agents.

## ✨ Key Features

*   **Code-First Development:** Define agents, tools, and orchestration logic directly in Python for maximum flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:**  Utilize pre-built tools, custom functions, and OpenAI specs, or seamlessly integrate existing tools for diverse agent capabilities and tight Google ecosystem integration.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible, interconnected hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how these technologies can work together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable ADK version using `pip`:

```bash
pip install google-adk
```

The release cadence is weekly.

### Development Version

Install directly from the `main` branch for the latest bug fixes and features:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may contain experimental changes or bugs. Use it for testing or accessing critical fixes before official releases.

## 📚 Documentation

Explore the full documentation for comprehensive guides:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

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

### Development UI

A built-in development UI for testing, evaluating, and debugging your agents.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome community contributions!  See the following for details:

*   [General contribution guidelines and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

The [llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt) files can be used as context for LLMs when developing agents using Vibe coding.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*