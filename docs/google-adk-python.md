# Agent Development Kit (ADK): Build Powerful AI Agents with Code

**Unleash the potential of AI agents with the Agent Development Kit (ADK), a code-first toolkit for building, evaluating, and deploying sophisticated AI agents.** ([Original Repo](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</div>

ADK is a flexible, modular, and model-agnostic framework, designed for building and deploying AI agents, particularly within the Google ecosystem.  It provides developers with the control and flexibility needed to create, deploy, and orchestrate agentic architectures ranging from simple to complex workflows.

**Key Features:**

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, OpenAPI specifications, and existing tools for diverse agent capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility and control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale with Vertex AI Agent Engine.

## 🤖 Agent-to-Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See the [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for usage.

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This is the recommended installation for most users. Release cadence is weekly.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version for the newest features and bug fixes. Be aware that it may contain experimental changes or bugs.

## 📚 Documentation

Comprehensive documentation is available:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Example: Defining and Using Agents

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

The built-in development UI simplifies testing, debugging, and showcasing agents:

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  Refer to these guidelines:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

Use the provided context files for LLM development: [llms.txt](./llms.txt) (summarized) and [llms-full.txt](./llms-full.txt) (full information).

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*