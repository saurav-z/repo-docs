![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful GenAI Applications with Ease

Griptape is a Python framework designed to simplify the development of generative AI (genAI) applications, offering flexible abstractions for LLMs, RAG, and more.

## Key Features

*   **Modular Architecture:** Build and orchestrate AI applications using Agents, Pipelines, and Workflows.
*   **Comprehensive Task Management:** Utilize Tasks as core building blocks for interaction with Engines, Tools, and other Griptape components.
*   **Advanced Memory Capabilities:** Leverage Conversation, Task, and Meta Memory for enhanced context and interaction.
*   **Versatile Drivers:** Integrate with diverse LLMs, retrieval systems, storage solutions, and external services through a comprehensive suite of Drivers.
*   **Powerful Engines:** Utilize specialized engines like RAG Engine, Extraction Engine, Summary Engine, and Eval Engine for specific use cases.
*   **Extensible Tooling:** Easily integrate and customize Tools to extend LLM capabilities and interact with data and services.
*   **Ready-to-Use Components:** Utilize Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers for enhanced LLM control and operation.

## Core Components

### 🏗️ Structures

*   🤖 **Agents**: Single Task configured for Agent-specific behavior.
*   🔄 **Pipelines**: Organizes a sequence of Tasks.
*   🌐 **Workflows**: Configure Tasks to operate in parallel.

### 📝 Tasks

Tasks are the core building blocks within Structures, enabling interaction with Engines, Tools, and other Griptape components.

### 🧠 Memory

*   💬 **Conversation Memory**: Retain and retrieve information across interactions.
*   🗃️ **Task Memory**: Keeps large or sensitive Task outputs off the prompt.
*   📊 **Meta Memory**: Pass in additional metadata to enhance context.

### 🚗 Drivers

Drivers facilitate interactions with external resources and services in Griptape. They allow you to swap out functionality and providers with minimal changes to your business logic.

#### LLM & Orchestration
*   🗣️ **Prompt Drivers**: Manage textual and image interactions with LLMs.
*   🤖 **Assistant Drivers**: Enable interactions with various “assistant” services.
*   📜 **Ruleset Drivers**: Load and apply rulesets from external sources.
*   🧠 **Conversation Memory Drivers**: Store and retrieve conversational data.
*   📡 **Event Listener Drivers**: Forward framework events to external services.
*   🏗️ **Structure Run Drivers**: Execute structures locally or in the cloud.

#### Retrieval & Storage
*   🔢 **Embedding Drivers**: Generate vector embeddings from textual inputs.
*   🔀 **Rerank Drivers**: Rerank search results for improved relevance.
*   💾 **Vector Store Drivers**: Manage the storage and retrieval of embeddings.
*   🗂️ **File Manager Drivers**: Handle file operations on local and remote storage.
*   💼 **SQL Drivers**: Interact with SQL databases.

#### Multimodal
*   🎨 **Image Generation Drivers**: Create images from text descriptions.
*   🗣️ **Text to Speech Drivers**: Convert text to speech.
*   🎙️ **Audio Transcription Drivers**: Convert audio to text.

#### Web
*   🔍 **Web Search Drivers**: Search the web for information.
*   🌐 **Web Scraper Drivers**: Extract data from web pages.

#### Observability
*   📈 **Observability Drivers**: Send trace and event data to observability platforms.

### 🔧 Tools

Tools provide capabilities for LLMs to interact with data and services. Griptape includes a variety of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/), and makes it easy to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### 🚂 Engines

Engines wrap Drivers and provide use-case-specific functionality:

*   📊 **RAG Engine**: Implements modular Retrieval Augmented Generation (RAG) pipelines.
*   🛠️ **Extraction Engine**: Extracts JSON or CSV data from unstructured text.
*   📝 **Summary Engine**: Generates summaries from textual content.
*   ✅ **Eval Engine**: Evaluates and scores the quality of generated text.

### 📦 Additional Components

*   📐 **Rulesets**: Steer LLM behavior.
*   🔄 **Loaders**: Load data from various sources.
*   🏺 **Artifacts**: Pass data of different types.
*   ✂️ **Chunkers**: Segment texts into manageable pieces.
*   🔢 **Tokenizers**: Count the number of tokens.

## Getting Started

For detailed information on installation and usage, please visit the [Griptape Documentation](https://docs.griptape.ai/).

Explore free online courses at [Griptape Trade School](https://learn.griptape.ai/).

## Examples

### Hello World

```python
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.rules import Rule
from griptape.tasks import PromptTask

task = PromptTask(
    prompt_driver=OpenAiChatPromptDriver(model="gpt-4.1"),
    rules=[Rule("Keep your answer to a few sentences.")],
)

result = task.run("How do I do a kickflip?")

print(result.value)
```

```text
To do a kickflip, start by positioning your front foot slightly angled near the middle of the board and your back foot on the tail.
Pop the tail down with your back foot while flicking the edge of the board with your front foot to make it spin.
Jump and keep your body centered over the board, then catch it with your feet and land smoothly. Practice and patience are key!
```

### Task and Workflow Example

```python
from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver
from griptape.drivers.web_search.duck_duck_go import DuckDuckGoWebSearchDriver
from griptape.rules import Rule, Ruleset
from griptape.structures import Workflow
from griptape.tasks import PromptTask, TextSummaryTask
from griptape.tools import WebScraperTool, WebSearchTool
from griptape.utils import StructureVisualizer
from pydantic import BaseModel


class Feature(BaseModel):
    name: str
    description: str
    emoji: str


class Output(BaseModel):
    answer: str
    key_features: list[Feature]


projects = ["griptape", "langchain", "crew-ai", "pydantic-ai"]

prompt_driver = OpenAiChatPromptDriver(model="gpt-4.1")
workflow = Workflow(
    tasks=[
        [
            PromptTask(
                id=f"project-{project}",
                input="Tell me about the open source project: {{ project }}.",
                prompt_driver=prompt_driver,
                context={"project": projects},
                output_schema=Output,
                tools=[
                    WebSearchTool(
                        web_search_driver=DuckDuckGoWebSearchDriver(),
                    ),
                    WebScraperTool(),
                ],
                child_ids=["summary"],
            )
            for project in projects
        ],
        TextSummaryTask(
            input="{{ parents_output_text }}",
            id="summary",
            rulesets=[
                Ruleset(
                    name="Format", rules=[Rule("Be detailed."), Rule("Include emojis.")]
                )
            ],
        ),
    ]
)

workflow.run()

print(StructureVisualizer(workflow).to_url())
```

```text
 Output: Here's a detailed summary of the open-source projects mentioned:

 1. **Griptape** 🛠️:                                                                                                            
    - Griptape is a modular Python framework designed for creating AI-powered applications. It focuses on securely connecting to
 enterprise data and APIs. The framework provides structured components like Agents, Pipelines, and Workflows, allowing for both
 parallel and sequential operations. It includes built-in tools and supports custom tool creation for data and service
 interaction.

 2. **LangChain** 🔗:
    - LangChain is a framework for building applications powered by Large Language Models (LLMs). It offers a standard interface
 for models, embeddings, and vector stores, facilitating real-time data augmentation and model interoperability. LangChain
 integrates with various data sources and external systems, making it adaptable to evolving technologies.

 3. **CrewAI** 🤖:
    - CrewAI is a standalone Python framework for orchestrating multi-agent AI systems. It allows developers to create and
 manage AI agents that collaborate on complex tasks. CrewAI emphasizes ease of use and scalability, providing tools and
 documentation to help developers build AI-powered solutions.

 4. **Pydantic-AI** 🧩:
    - Pydantic-AI is a Python agent framework that simplifies the development of production-grade applications with Generative
 AI. Built on Pydantic, it supports various AI models and provides features like type-safe design, structured response
 validation, and dependency injection. Pydantic-AI aims to bring the ease of FastAPI development to AI applications.

 These projects offer diverse tools and frameworks for developing AI applications, each with unique features and capabilities
 tailored to different aspects of AI development.
```

```mermaid
    graph TD;
    griptape-->summary;
    langchain-->summary;
    pydantic-ai-->summary;
    crew-ai-->summary;
```

## Versioning

Griptape uses [Semantic Versioning](https://semver.org/).

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before you start.

## License

Griptape is available under the Apache 2.0 License.

[Back to Top](https://github.com/griptape-ai/griptape)