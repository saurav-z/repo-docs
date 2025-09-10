# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

[<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>](https://github.com/google/adk-python)

**ADK empowers developers to build, evaluate, and deploy sophisticated AI agents with a code-first, flexible, and modular Python toolkit.** This open-source framework, optimized for Gemini and the Google ecosystem, provides the tools you need to create intelligent agents that can handle complex tasks.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

## Key Features:

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for maximum flexibility, testability, and version control.
*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, OpenAPI specs, or integrate with existing tools to equip your agents with diverse capabilities and tight integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable and complex applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without writing any code.

## What's New:

*   **[Agent Config](https://google.github.io/adk-docs/agents/config/):** Quickly create agents without writing any code.

## 🤖 Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  Explore an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how they can work together.

## 🚀 Getting Started

### Installation

Install the Agent Development Kit using `pip`:

```bash
pip install google-adk
```

For the latest features and bug fixes, install from the main branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## 📚 Documentation and Resources

*   **[Documentation](https://google.github.io/adk-docs):** Comprehensive guides for building, evaluating, and deploying agents.
*   **[Samples](https://github.com/google/adk-samples):** Explore code examples to get started quickly.
*   **[Java ADK](https://github.com/google/adk-java):** The Java version of ADK.
*   **[ADK Web](https://github.com/google/adk-web):** Explore the ADK Web app.

## 🏁 Code Examples

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

Test, evaluate, debug, and showcase your agents with the built-in development UI.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI Example"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome contributions!  Review the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## Vibe Coding

When developing agents with vibe coding, use the context provided in [llms.txt](./llms.txt) (summarized) or [llms-full.txt](./llms-full.txt) (full information).

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.