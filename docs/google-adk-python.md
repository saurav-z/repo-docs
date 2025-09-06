<!--
  SPDX-License-Identifier: Apache-2.0
-->

# Agent Development Kit (ADK): Build Powerful AI Agents with Code

**Unleash the power of code to build, evaluate, and deploy sophisticated AI agents with the Agent Development Kit (ADK) from Google.** ([Original Repository](https://github.com/google/adk-python))

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<p align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo">
</p>

ADK is a powerful, open-source Python toolkit designed to simplify and accelerate the development of AI agents. It offers a code-first approach, providing developers with maximum flexibility and control over agent creation, deployment, and orchestration.  While optimized for the Google ecosystem (including Gemini), ADK is designed to be model-agnostic and deployment-agnostic, ensuring compatibility with other frameworks.

**Key Features of ADK:**

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python, enabling seamless versioning, testing, and customization.
*   **Rich Tool Ecosystem:** Leverage pre-built tools, custom functions, and OpenAPI specs, or integrate existing tools to provide agents with diverse capabilities.
*   **Modular Multi-Agent Systems:** Build scalable applications by composing multiple specialized agents into flexible, interconnected hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale them with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code. Check out the
  [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.

**ADK simplifies complex agent development tasks:**

*   **Define a single agent:**

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

*   **Define a multi-agent system:**

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

*   **Development UI:** A built-in development UI to help you test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI">

*   **Evaluate Agents**

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Agent2Agent (A2A) Protocol Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication. Explore this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see A2A and ADK working together.

## Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using pip:

```bash
pip install google-adk
```

### Development Version

Install directly from the main branch for the latest features and bug fixes:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

## Documentation

For in-depth information, tutorials, and API references, consult the official documentation:

*   **[Documentation](https://google.github.io/adk-docs)**

## Contributing

Contributions are welcome! Review the [contribution guidelines](https://google.github.io/adk-docs/contributing-guide/) to get started.  Code contributions should follow the [Code Contributing Guidelines](./CONTRIBUTING.md).

## Vibe Coding

For agent development using vibe coding, use the provided context files:

*   `llms.txt`: Summarized information for your LLM.
*   `llms-full.txt`: Full information for LLMs with large context windows.

## License

This project is licensed under the Apache 2.0 License.  See the [LICENSE](LICENSE) file for details.

---

*Build your AI agents with ADK!*