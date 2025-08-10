![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Advanced AI Applications with Ease

Griptape is a Python framework designed to streamline the development of powerful and flexible Generative AI (genAI) applications.  Leverage LLMs, RAG, and more to build sophisticated AI solutions.

## Key Features

*   **Modular Structure:**  Build applications with Agents, Pipelines, and Workflows.
*   **Intuitive Tasks:** Core building blocks for interacting with Engines, Tools, and other components.
*   **Flexible Memory:**  Implement Conversation, Task, and Meta memory for enhanced context.
*   **Versatile Drivers:** Connect to external resources and services via LLM, Retrieval, Multimodal, Web, and Observability drivers.
*   **Extensible Tools:** Leverage built-in tools or easily create custom tools.
*   **Powerful Engines:** Utilize RAG, Extraction, Summary, and Eval Engines for specific use cases.
*   **Additional Components:** Utilize Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers for complete control.

## Core Components Explained

### 🏗️ Structures

*   🤖 **Agents:** Single-task structures for agent-specific behaviors.
*   🔄 **Pipelines:** Organize sequential task execution.
*   🌐 **Workflows:** Enable parallel task execution.

### 📝 Tasks

*   The fundamental building blocks within Structures.

### 🧠 Memory

*   💬 **Conversation Memory:** Retain information across interactions.
*   🗃️ **Task Memory:** Keeps outputs off the prompt.
*   📊 **Meta Memory:** Enhance context with additional metadata.

### 🚗 Drivers

*   Facilitate interaction with external resources.

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

*   Provide capabilities for LLMs to interact with data and services. Explore [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/) and learn how to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### 🚂 Engines

*   Wrap Drivers and provide use-case-specific functionality:
    *   📊 **RAG Engine** - Retrieval Augmented Generation
    *   🛠️ **Extraction Engine** - Extract JSON or CSV
    *   📝 **Summary Engine** - Generate summaries
    *   ✅ **Eval Engine** - Evaluate generated text

### 📦 Additional Components

*   📐 **Rulesets:** Steer LLM behavior.
*   🔄 **Loaders:** Load data from various sources.
*   🏺 **Artifacts:** Pass data of different types.
*   ✂️ **Chunkers:** Segment texts into pieces.
*   🔢 **Tokenizers:** Count tokens.

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

### Task and Workflow

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

## Documentation

For comprehensive information on installation, usage, and more, please visit the [Griptape documentation](https://docs.griptape.ai/).  Check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Versioning

Griptape follows [Semantic Versioning](https://semver.org/).

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md).

## License

Griptape is licensed under the Apache 2.0 License.

## Get Started

Explore the [Griptape GitHub repository](https://github.com/griptape-ai/griptape) to get started building your next AI application.