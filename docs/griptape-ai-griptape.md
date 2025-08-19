# Griptape: Build Powerful Generative AI Applications with Ease

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

**Griptape is a Python framework that simplifies building sophisticated generative AI applications, offering flexible abstractions for LLMs, RAG, and more.**

## Key Features

*   **Modular Structures:** Build AI applications with Agents, Pipelines, and Workflows for structured task management.
*   **Flexible Task Management:** Design and execute tasks using various Engines, Tools, and Memory components.
*   **Comprehensive Memory Solutions:** Implement Conversation, Task, and Meta Memory to retain and enrich context.
*   **Extensive Driver Support:** Seamlessly integrate with various LLMs, databases, web services, and storage solutions.
*   **Rich Tool Ecosystem:** Leverage built-in tools and easily create custom tools to enhance LLM capabilities.
*   **Powerful Engines:** Utilize RAG, Extraction, Summary, and Eval Engines for advanced use cases.
*   **Additional Components:** Leverage Rulesets, Loaders, Artifacts, and Chunkers to optimize application performance.

## Core Components

### Structures

*   **Agents:** Single Task, configured for agent-specific behavior.
*   **Pipelines:** Organizes a sequence of tasks.
*   **Workflows:** Configures tasks to operate in parallel.

### Tasks

Tasks are the core building blocks within Structures, enabling interaction with Engines, Tools, and other Griptape components.

### Memory

*   **Conversation Memory:** Retains information across interactions.
*   **Task Memory:** Keeps large or sensitive Task outputs off the prompt.
*   **Meta Memory:** Adds extra metadata to enhance context.

### Drivers

Facilitate interactions with external resources and services, allowing you to swap out functionality and providers with minimal changes to your business logic.

#### LLM & Orchestration

*   **Prompt Drivers:** Manage textual and image interactions with LLMs.
*   **Assistant Drivers:** Enable interactions with various "assistant" services.
*   **Ruleset Drivers:** Load and apply rulesets from external sources.
*   **Conversation Memory Drivers:** Store and retrieve conversational data.
*   **Event Listener Drivers:** Forward framework events to external services.
*   **Structure Run Drivers:** Execute structures locally or in the cloud.

#### Retrieval & Storage

*   **Embedding Drivers:** Generate vector embeddings from textual inputs.
*   **Rerank Drivers:** Rerank search results for improved relevance.
*   **Vector Store Drivers:** Manage the storage and retrieval of embeddings.
*   **File Manager Drivers:** Handle file operations on local and remote storage.
*   **SQL Drivers:** Interact with SQL databases.

#### Multimodal

*   **Image Generation Drivers:** Create images from text descriptions.
*   **Text to Speech Drivers:** Convert text to speech.
*   **Audio Transcription Drivers:** Convert audio to text.

#### Web

*   **Web Search Drivers:** Search the web for information.
*   **Web Scraper Drivers:** Extract data from web pages.

#### Observability

*   **Observability Drivers:** Send trace and event data to observability platforms.

### Tools

Tools provide capabilities for LLMs to interact with data and services.
Griptape includes a variety of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/), and makes it easy to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### Engines

Engines wrap Drivers and provide use-case-specific functionality:

*   **RAG Engine:** For implementing Retrieval Augmented Generation (RAG) pipelines.
*   **Extraction Engine:** Extracts JSON or CSV data from unstructured text.
*   **Summary Engine:** Generates summaries from textual content.
*   **Eval Engine:** Evaluates and scores the quality of generated text.

### Additional Components

*   **Rulesets:** Steer LLM behavior with minimal prompt engineering.
*   **Loaders:** Load data from various sources.
*   **Artifacts:** Pass data of different types between components.
*   **Chunkers:** Segment texts into manageable pieces.
*   **Tokenizers:** Count the number of tokens to respect LLM limits.

## Documentation

For detailed information on installation, usage, and more, please visit the [Griptape Documentation](https://docs.griptape.ai/).

Explore [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Examples

### Hello World Example

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

Output:

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

Output:

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

[Back to Top](#griptape-build-powerful-generative-ai-applications-with-ease) - [Griptape GitHub Repository](https://github.com/griptape-ai/griptape)