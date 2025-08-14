# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unleash the power of AI agents with the Agent Development Kit (ADK), a flexible and code-first Python toolkit to build, evaluate, and deploy cutting-edge AI agents.**

[<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="ADK Logo" align="right">](https://github.com/google/adk-python)

ADK empowers developers to create sophisticated AI agents, offering unparalleled flexibility and control. Built with a code-first approach and optimized for the Google ecosystem, ADK seamlessly integrates with other frameworks and is designed for agentic architectures ranging from simple tasks to complex workflows.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

**Key Features:**

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, enabling seamless version control, testing, and ultimate flexibility.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specs to give agents diverse capabilities, while integrating with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies for complex tasks.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agents with a user-friendly development UI.
*   **Agent2Agent (A2A) Protocol Integration:**  Seamlessly communicate between agents using the A2A protocol.
*   **Evaluation Tools:** Built-in evaluation capabilities to assess agent performance.

**Important Links:**

*   [Documentation](https://google.github.io/adk-docs/)
*   [Samples](https://github.com/google/adk-samples)
*   [Java ADK](https://github.com/google/adk-java)
*   [ADK Web](https://github.com/google/adk-web)

---

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version from PyPI:

```bash
pip install google-adk
```

We release weekly updates.

### Development Version

Install the latest features and bug fixes from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Note: Use the development version for testing or accessing critical fixes, keeping in mind it may include experimental changes or bugs.

---

## 📚 Getting Started

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

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="Development UI Example">

###  Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

---

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for practical implementation details.

---

## 🤝 Contributing

We welcome contributions! Check out our [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md).

---

## Vibe Coding

Use [llms.txt](./llms.txt) (summarized) and [llms-full.txt](./llms-full.txt) as context for LLMs in Vibe coding.

---

## 📄 License

Licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

**Explore the future of AI agents with ADK!**

**[View the source code on GitHub](https://github.com/google/adk-python)**