# Agent Development Kit (ADK): Build Powerful AI Agents with Python

**ADK empowers developers to build, evaluate, and deploy sophisticated AI agents with flexibility and control using a code-first Python approach. [(Original Repository)](https://github.com/google/adk-python)**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<div align="center">
    <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256" alt="Agent Development Kit Logo"/>
</div>

**ADK is your flexible, modular, and model-agnostic toolkit for building AI agents.** Optimized for the Google ecosystem but compatible with others, ADK transforms agent development into a streamlined software development experience.  Create, deploy, and orchestrate agents, from simple tasks to complex workflows, with ease.

---

## Key Features:

*   **Rich Tool Ecosystem:** Integrate pre-built tools, custom functions, and OpenAPI specs. ADK seamlessly integrates with the Google ecosystem.
*   **Code-First Development:**  Define agent logic, tools, and orchestration directly in Python for maximum flexibility and version control.
*   **Modular Multi-Agent Systems:** Design scalable applications by composing multiple specialized agents into flexible hierarchies.
*   **Deploy Anywhere:** Easily containerize and deploy agents on Cloud Run or scale seamlessly with Vertex AI Agent Engine.

## 🤖 Agent-to-Agent (A2A) Protocol Integration

ADK integrates with the [A2A protocol](https://github.com/google-a2a/A2A/) for advanced remote agent communication.  See an [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents) of how they work together.

## 🚀 Installation

### Stable Release (Recommended)

```bash
pip install google-adk
```

*   This version represents the most recent, officially released version of ADK. Weekly releases are planned.

### Development Version

```bash
pip install git+https://github.com/google/adk-python.git@main
```

*   Access the latest bug fixes and features merged into the main branch.  Use this for testing or critical fixes, but be aware it may contain experimental changes.

## 📚 Documentation

Comprehensive guides for building, evaluating, and deploying agents can be found in the official documentation:

*   **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlights

### Define a Single Agent

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

### Define a Multi-Agent System

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

ADK provides a built-in development UI to test, evaluate, debug, and showcase your agent(s).

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png" alt="ADK Development UI"/>

### Evaluate Agents

```bash
adk eval \
    samples_for_testing/hello_world \
    samples_for_testing/hello_world/hello_world_eval_set_001.evalset.json
```

## 🤝 Contributing

We encourage community contributions!  See the following resources for guidance:

*   [General contribution guideline and flow](https://google.github.io/adk-docs/contributing-guide/).
*   [Code Contributing Guidelines](./CONTRIBUTING.md)

## Vibe Coding

The provided context files, [llms.txt](./llms.txt) (summarized) and [llms-full.txt](./llms-full.txt) (full), can be used to provide context to LLMs.  Use the version that fits your LLM's context window.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
```
Key improvements and SEO considerations:

*   **Clear Title and Hook:**  The title and first sentence are concise and include relevant keywords (Agent Development Kit, AI Agents, Python).
*   **Keyword Optimization:**  Keywords like "AI agents," "Python," "toolkit," "build," "evaluate," "deploy," "flexible," "modular," and "Google ecosystem" are strategically placed.
*   **Subheadings:**  Uses clear, descriptive subheadings to break up the content and improve readability.
*   **Bulleted Lists:**  Highlights key features in a clear, easy-to-scan format.
*   **Alt Text for Images:**  Added `alt` text to the images for accessibility and SEO.
*   **Concise Language:**  Uses direct and active language to keep the reader engaged.
*   **Call to Action (Implied):** The entire README acts as a call to action, encouraging users to explore and use the ADK.
*   **Internal Linking:** Added a link to the original repo and other helpful documentation.