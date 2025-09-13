# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Unleash the power of AI agents with Google's Agent Development Kit (ADK), a code-first toolkit designed for flexibility and control.**

[Link to Original Repo](https://github.com/google/adk-python)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

[<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>](https://github.com/google/adk-python)

ADK is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents. It's designed for developers who want ultimate flexibility and control over their agentic architectures, from simple tasks to complex workflows. While optimized for the Google ecosystem (Gemini), ADK is model-agnostic and deployment-agnostic, making it compatible with various frameworks.

**Key Features:**

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs to equip your agents with diverse capabilities, with tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for enhanced flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without writing any code, using the Agent Config feature.
*   **Tool Confirmation:** Implement a tool confirmation flow (HITL) to ensure accurate tool execution with confirmation and custom input.
*   **Development UI:** Test, evaluate, debug, and showcase your agents using the built-in development UI.

## 🔥 What's New

*   **Agent Config**: Build agents without code. Check out the
    [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

*   **Tool Confirmation**: A [tool confirmation flow(HITL)](https://google.github.io/adk-docs/tools/confirmation/) that can guard tool execution with explicit confirmation and custom input

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. Explore an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how to use A2A with ADK.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using pip:

```bash
pip install google-adk
```

The release cadence is roughly bi-weekly.

### Development Version

Get the latest features and bug fixes by installing directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Note: The development version may include experimental features or bugs. Use with caution.

## 📚 Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Examples

### Define a single agent:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.5-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a multi-agent system:

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="greeter", model="gemini-2.5-flash", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.5-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash",
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

## 🤝 Contributing

We welcome contributions! See our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/)
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

For agent development via vibe coding, use [llms.txt](./llms.txt) or [llms-full.txt](./llms-full.txt) as context to your LLM.

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*