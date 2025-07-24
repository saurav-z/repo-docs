# Agent Development Kit (ADK): Build, Evaluate, and Deploy AI Agents with Ease

**Unlock the power of AI agents with the Agent Development Kit (ADK), a code-first Python toolkit designed for flexible and efficient agent creation, evaluation, and deployment.** ([Original Repo](https://github.com/google/adk-python))

<p align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</p>

The Agent Development Kit (ADK) empowers developers to build sophisticated AI agents with control and flexibility. Optimized for the Google ecosystem and built for compatibility with other frameworks, ADK simplifies the development process, from simple tasks to complex, multi-agent workflows.

## Key Features:

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for full flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, OpenAPI specs, or integrate existing tools to give agents diverse capabilities.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Built-in Development UI:** Test, evaluate, debug, and showcase your agent(s) with the included UI.
*   **Model-Agnostic:** ADK is designed to work with various language models.

## Get Started:

### Installation

**Stable Release (Recommended):**

```bash
pip install google-adk
```

**Development Version:** (For the latest features and bug fixes)

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## Example Code:

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

### Define a Multi-Agent System:

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

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Documentation:

*   [Documentation](https://google.github.io/adk-docs)

## Contributing:

We welcome community contributions!  See the [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/) and [Code Contributing Guidelines](./CONTRIBUTING.md) for details.

## License:

This project is licensed under the Apache 2.0 License.  See the [LICENSE](LICENSE) file for details.