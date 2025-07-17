# Agent Development Kit (ADK) - Build, Evaluate, and Deploy Powerful AI Agents

**Quickly build and deploy sophisticated AI agents with the Google Agent Development Kit (ADK), offering flexibility, control, and seamless integration with the Google ecosystem.  [Learn more at the original repository](https://github.com/google/adk-python).**

ADK is an open-source, code-first Python toolkit designed to empower developers to build, evaluate, and deploy advanced AI agents.  This framework provides a modular and flexible approach to agent development, optimized for Gemini and the Google ecosystem but built to be model-agnostic and deployment-agnostic for compatibility across platforms.

## Key Features

*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specs for diverse agent capabilities, with tight Google ecosystem integration.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for enhanced flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Getting Started

### Installation

#### Stable Release (Recommended)

Install the latest stable release via pip:

```bash
pip install google-adk
```

This is the recommended option for most users.

#### Development Version

Access the latest bug fixes and features by installing directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version for testing upcoming features or accessing critical fixes.

## Code Examples

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

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Agent2Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they work together.

## Documentation and Resources

*   [Documentation](https://google.github.io/adk-docs)
*   [Samples](https://github.com/google/adk-samples)
*   [Java ADK](https://github.com/google/adk-java)
*   [ADK Web](https://github.com/google/adk-web)

## Contributing

We welcome community contributions!  Please review the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and the [Code Contributing Guidelines](./CONTRIBUTING.md) to get started.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.