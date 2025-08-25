![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful GenAI Applications with Ease

**Griptape is the Python framework that empowers developers to build sophisticated Generative AI applications with modularity and efficiency.** [Explore the Griptape Repository](https://github.com/griptape-ai/griptape).

## Key Features

*   **Modular Architecture:** Easily build and manage complex AI workflows with Agents, Pipelines, and Workflows.
*   **Simplified LLM Interaction:**  Seamlessly integrate and manage interactions with Large Language Models (LLMs).
*   **Memory Management:** Utilize Conversation, Task, and Meta Memory for richer, more context-aware interactions.
*   **Extensive Driver Support:**  Connect to a wide range of external resources and services through flexible drivers for LLMs, Retrieval & Storage, Multimodal, Web, and Observability.
*   **Built-in Tools & Customization:** Leverage a rich library of built-in tools and create custom tools to enhance your AI applications.
*   **Use-Case-Specific Engines:** Accelerate development with pre-built Engines like RAG, Extraction, Summary, and Eval.
*   **Comprehensive Components:** Utilize Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers for advanced control and flexibility.

## Core Components

### Structures

*   **Agents:** Single-task configurations for specialized agent behavior.
*   **Pipelines:** Organize a sequence of tasks, passing outputs between them.
*   **Workflows:** Configure tasks to operate concurrently.

### Tasks

*   The fundamental building blocks for interacting with Engines, Tools, and other Griptape components.

### Memory

*   **Conversation Memory:** Preserve context across interactions.
*   **Task Memory:** Store large outputs to reduce prompt size and increase security.
*   **Meta Memory:** Provide additional metadata to LLMs.

### Drivers

Facilitate interaction with external resources and services, enabling easy swapping of functionality and providers.

#### LLM & Orchestration

*   **Prompt Drivers:** Manage textual and image interactions with LLMs.
*   **Assistant Drivers:** Enable interactions with various “assistant” services.
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

*   Tools provide capabilities for LLMs to interact with data and services.
    *   See [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/).
    *   Learn how to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### Engines

Engines wrap Drivers and provide use-case-specific functionality:

*   **RAG Engine:** Implement modular Retrieval Augmented Generation (RAG) pipelines.
*   **Extraction Engine:** Extract structured data (JSON, CSV) from text.
*   **Summary Engine:** Generate summaries from textual content.
*   **Eval Engine:** Evaluate and score the quality of generated text.

### Additional Components

*   **Rulesets:** Control LLM behavior with minimal prompt engineering.
*   **Loaders:** Load data from various sources.
*   **Artifacts:** Pass data between Griptape components.
*   **Chunkers:** Segment text for efficient processing.
*   **Tokenizers:** Manage token limits for LLMs.

## Documentation

Find comprehensive information on installation and usage in the [Griptape documentation](https://docs.griptape.ai/).

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

Review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) to contribute to Griptape!

## License

Griptape is licensed under the Apache 2.0 License.