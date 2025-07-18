# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**ADK empowers developers to create, evaluate, and deploy sophisticated AI agents with unparalleled flexibility and control, making agent development feel like software development.** [See the original repo](https://github.com/google/adk-python).

## Key Features of ADK:

*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specifications to give your agents diverse capabilities, with seamless integration with the Google ecosystem.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## ADK Integration with Agent2Agent (A2A) Protocol

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  Explore this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they can work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

This version is updated weekly and is recommended for most users.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

Use the development version for testing the latest features and bug fixes, but be aware that it may contain experimental changes.

## Documentation

*   **[Documentation](https://google.github.io/adk-docs)**: Explore detailed guides for building, evaluating, and deploying agents.

## Feature Highlight

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

ADK provides a built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Contributing

We welcome contributions!  Please see our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.