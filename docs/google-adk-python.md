# Agent Development Kit (ADK): Build, Evaluate, and Deploy Powerful AI Agents

ADK is an open-source, code-first Python toolkit from Google for building sophisticated AI agents with flexibility and control, empowering developers to create advanced agentic architectures. [Explore the original repository](https://github.com/google/adk-python).

## Key Features of the Agent Development Kit

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications for diverse agent capabilities, seamlessly integrated with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, ensuring flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Getting Started

### Installation

**Stable Release (Recommended):**

```bash
pip install google-adk
```

This version is updated weekly and recommended for most users.

**Development Version:**

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Install from the `main` branch for access to the newest features and bug fixes before they are officially released. This version may contain experimental changes or bugs.

## Core Concepts and Examples

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

### Build a Multi-Agent System:

Define a multi-agent system with a coordinator agent, greeter agent, and task execution agent:

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

ADK offers a built-in development UI for testing, evaluating, and debugging your agents:

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Documentation

*   [Documentation](https://google.github.io/adk-docs) for detailed guides on building, evaluating, and deploying agents.

## Contributing

We welcome contributions from the community! See our:
*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)