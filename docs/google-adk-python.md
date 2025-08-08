# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unlock the potential of sophisticated AI agents with Google's open-source Agent Development Kit (ADK) for Python.** ([Original Repository](https://github.com/google/adk-python))

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
</div>

ADK is a code-first Python toolkit designed for building, evaluating, and deploying AI agents, offering flexibility and control throughout the development lifecycle. Built for the Google ecosystem but model-agnostic, you can create agentic architectures ranging from simple tasks to complex workflows.

## Key Features

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specifications, or utilize existing tools to enhance agent capabilities.
*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for seamless flexibility, easy testing, and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## Core Functionality:

*   **Agent Definition:** Create a single agent with ease:
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

*   **Multi-Agent Systems:** Design systems with a coordinator and sub-agents:

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

*   **Evaluation:** Evaluate agents with the CLI:
    ```bash
    adk eval \
        samples_for_testing/hello_world \
        samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
    ```

*   **Development UI:** A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

    <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

## Agent2Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) for how they can work together.

## Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## Documentation

*   **[Documentation](https://google.github.io/adk-docs)**

## Contributing

We welcome contributions! Please refer to the following for contribution guidelines:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

The `llms.txt` and `llms-full.txt` files can be used as context for LLMs when developing agents via vibe coding.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.