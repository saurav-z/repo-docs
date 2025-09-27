# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**Unlock the power of AI agents with Google's Agent Development Kit (ADK), a flexible and code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents.**  

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
  <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

<div align="center">
  <p>An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.</p>
  <p>
    <a href="https://google.github.io/adk-docs/">Docs</a> |
    <a href="https://github.com/google/adk-samples">Samples</a> |
    <a href="https://github.com/google/adk-java">Java ADK</a> |
    <a href="https://github.com/google/adk-web">ADK Web</a>
  </p>
</div>

ADK is a modular and flexible framework designed to streamline the development and deployment of AI agents. While optimized for the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and compatible with other frameworks. Designed to make agent development feel more like traditional software development, ADK simplifies the creation, deployment, and orchestration of agentic architectures, ranging from simple tasks to complex workflows.

## ✨ Key Features

*   **Code-First Development:** Define agent logic, tools, and orchestration directly in Python for ultimate flexibility, testability, and versioning.
*   **Rich Tool Ecosystem:** Utilize pre-built tools, custom functions, and OpenAPI specs to give agents diverse capabilities, ensuring seamless integration with the Google ecosystem.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.
*   **Agent Config:** Build agents without code, utilizing the [Agent Config](https://google.github.io/adk-docs/agents/config/) feature.
*   **Tool Confirmation:** Implement a [tool confirmation flow (HITL)](https://google.github.io/adk-docs/tools/confirmation/) to guard tool execution with explicit confirmation and custom input.

## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

ADK seamlessly integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for remote agent-to-agent communication.  Explore the [A2A samples](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) to see how they can be used together.

## 🚀 Installation

### Stable Release (Recommended)

Install the latest stable version of ADK using `pip`:

```bash
pip install google-adk
```

The release cadence is approximately bi-weekly.

This version is recommended for most users as it represents the most recent official release.

### Development Version

For access to the latest bug fixes and features before official releases, install directly from the `main` branch:

```bash
pip install git+https://github.com/google/adk-python.git@main
```

**Note:** The development version may include experimental changes or bugs. Use it primarily for testing upcoming changes or accessing critical fixes before they are officially released.

## 📚 Documentation

Comprehensive documentation is available to guide you through building, evaluating, and deploying agents:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

### Define a Single Agent

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.5-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a Multi-Agent System

Define a multi-agent system with coordinator agent, greeter agent, and task execution agent. Then ADK engine and the model will guide the agents works together to accomplish the task.

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="greeter", model="gemini-2.5-flash", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.5-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.5-flash",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

A built-in development UI is available to help you test, evaluate, debug, and showcase your agents.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## Vibe Coding

For agent development using vibe coding, use the context provided in [llms.txt](./llms.txt) (summarized) and [llms-full.txt](./llms-full.txt) (full) for your LLM.

## 🤝 Contributing

Contributions are welcome!  Please see our:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md) for code contributions.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*

[Back to the Top](https://github.com/google/adk-python)
```
Key improvements and SEO considerations:

*   **Strong Title and Hook:**  The title is optimized with the key phrase "Agent Development Kit" and a one-sentence hook to immediately capture attention.
*   **Clear Headings:**  Organized with clear headings for better readability and SEO structure.
*   **Bulleted Key Features:**  Emphasizes core benefits and features using bullet points, making them easy to scan.
*   **Keyword Optimization:**  Includes relevant keywords like "AI agents," "Python," "toolkit," "build," "evaluate," "deploy," "Gemini," and "Google ecosystem."
*   **Concise Language:**  Streamlines the descriptions for clarity.
*   **Call to Action:**  "Happy Agent Building!" encourages engagement.
*   **Internal Linking:** Links to relevant documentation.
*   **External Links:** Adds links for easy access to repo and important documentations.
*   **Alt Text for Images:** Includes alt text for image accessibility and SEO.
*   **Back to Top Link:** Added a link to get users back to the top of the README.