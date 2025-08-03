# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Unleash the power of AI agents with Google's Agent Development Kit (ADK), a flexible, code-first Python toolkit.**  ([See the original repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</div>

ADK empowers developers to create sophisticated AI agents for diverse applications, from simple tasks to complex workflows.  Built for flexibility, ADK is model-agnostic and deployment-agnostic, ensuring compatibility and control.

**Key Features:**

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications for diverse agent capabilities, with tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with a dedicated UI.
*   **Agent-to-Agent (A2A) Protocol Integration:** Leverage the A2A protocol for remote agent communication.

---

## 🚀 Getting Started

### Installation

**Stable Release (Recommended):**

```bash
pip install google-adk
```

The release cadence is weekly. This version is recommended for most users as it represents the most recent official release.

**Development Version:**

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*Note: The development version may contain experimental features or bugs. Use it for testing or accessing the latest changes.*

## 📚 Documentation and Resources

*   **[Documentation](https://google.github.io/adk-docs)**: Comprehensive guides for building, evaluating, and deploying agents.
*   [Samples](https://github.com/google/adk-samples)
*   [Java ADK](https://github.com/google/adk-java)
*   [ADK Web](https://github.com/google/adk-web)
*   [A2A protocol](https://github.com/google-a2a/A2A/)

## 🧑‍💻 Core Examples

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

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome community contributions! See the following resources to get involved:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 🧠 Vibe Coding

Utilize the provided context files ([llms.txt](./llms.txt) and [llms-full.txt](./llms-full.txt)) to aid in agent development through vibe coding, providing context for LLMs.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*