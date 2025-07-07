# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Empower your development of sophisticated AI agents with the Agent Development Kit (ADK), a code-first Python toolkit designed for flexibility and control. [Explore the original repository](https://github.com/google/adk-python) for more information.**

ADK is a flexible and modular framework for developing and deploying AI agents. Optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and compatible with other frameworks. ADK allows developers to easily create, deploy, and orchestrate agentic architectures, from simple tasks to complex workflows.

## Key Features:

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs, for tight integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## 📚 Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents.

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

### Single Agent Definition

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

### Multi-Agent System

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

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We welcome community contributions. See our:
*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---