![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful Generative AI Applications with Ease

Griptape is a Python framework designed to streamline the development of sophisticated generative AI applications. Explore the [Griptape GitHub repository](https://github.com/griptape-ai/griptape) for more details.

## Key Features

*   **Modular Architecture:** Build applications with Agents, Pipelines, and Workflows for structured AI operations.
*   **LLM Integration:** Seamlessly interact with Large Language Models (LLMs) through prompt and assistant drivers.
*   **Memory Management:** Implement conversation, task, and meta memory to enhance context and relevance.
*   **Extensive Driver Support:** Connect to external resources and services with a wide range of drivers.
    *   **LLM & Orchestration Drivers:** Prompt, Assistant, Ruleset, Memory, Event Listener, and Structure Run Drivers.
    *   **Retrieval & Storage Drivers:** Embedding, Rerank, Vector Store, File Manager, and SQL Drivers.
    *   **Multimodal Drivers:** Image Generation, Text to Speech, and Audio Transcription Drivers.
    *   **Web Drivers:** Web Search and Web Scraper Drivers.
    *   **Observability Drivers:** Observability drivers for sending trace and event data to observability platforms.
*   **Flexible Tooling:** Utilize built-in and custom tools for data interaction and service integration.
*   **Powerful Engines:** Leverage specialized engines such as RAG, Extraction, Summary, and Eval engines.
*   **Additional Components:** Take advantage of Rulesets, Loaders, Artifacts, and Chunkers for comprehensive application development.

## Core Components

### Structures

*   **Agents:** Designed for Agent-specific behavior with a single Task.
*   **Pipelines:** Organize a sequence of Tasks for sequential operations.
*   **Workflows:** Configure Tasks to run in parallel.

### Tasks

Tasks are the core building blocks within Structures, enabling interaction with Engines, Tools, and other Griptape components.

### Memory

*   **Conversation Memory:** Enables LLMs to retain and retrieve information across interactions.
*   **Task Memory:** Keeps large or sensitive Task outputs off the prompt that is sent to the LLM.
*   **Meta Memory:** Enables passing in additional metadata to the LLM, enhancing the context and relevance of the interaction.

### Drivers

Drivers facilitate interactions with external resources and services in Griptape. They allow you to swap out functionality and providers with minimal changes to your business logic.

### Tools

Tools provide capabilities for LLMs to interact with data and services. Griptape includes a variety of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/), and makes it easy to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### Engines

Engines wrap Drivers and provide use-case-specific functionality:

*   **RAG Engine** is an abstraction for implementing modular Retrieval Augmented Generation (RAG) pipelines.
*   **Extraction Engine** extracts JSON or CSV data from unstructured text.
*   **Summary Engine** generates summaries from textual content.
*   **Eval Engine** evaluates and scores the quality of generated text.

### Additional Components

*   **Rulesets** steer LLM behavior with minimal prompt engineering.
*   **Loaders** load data from various sources.
*   **Artifacts** allow for passing data of different types between Griptape components.
*   **Chunkers** segment texts into manageable pieces for diverse text types.
*   **Tokenizers** count the number of tokens in a text to not exceed LLM token limits.

## Documentation

Comprehensive documentation is available at [https://docs.griptape.ai/](https://docs.griptape.ai/).

Explore [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Hello World Example

Here's a minimal example of griptape:

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

## Task and Workflow Example

Here is a concise example using griptape to research open source projects:

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

We welcome contributions! Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md).

## License

Griptape is available under the Apache 2.0 License.