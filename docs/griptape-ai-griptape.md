# Griptape: Build Powerful GenAI Applications with Ease

[<img src="https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg" alt="Griptape Logo" width="200">](https://github.com/griptape-ai/griptape)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

Griptape is a powerful Python framework that empowers developers to build robust and scalable Generative AI (GenAI) applications.

**Key Features:**

*   **Modular Architecture:** Build applications using intuitive Structures, Tasks, and Engines.
*   **Flexible Abstractions:** Easily work with Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and more.
*   **Extensive Driver Support:** Integrate with various LLMs, web services, databases, and storage solutions.
*   **Built-in Tools:** Access a wide range of pre-built tools and create custom ones.
*   **Memory Management:** Implement robust conversation, task, and meta memory features.

## Core Components

### Structures

*   **Agents:** Single Task configured for specific behaviors.
*   **Pipelines:** Organize Tasks in a sequence.
*   **Workflows:** Configure Tasks for parallel execution.

### Tasks

The fundamental building blocks for interacting with Engines, Tools, and other Griptape components.

### Memory

*   **Conversation Memory:** Maintain context across interactions.
*   **Task Memory:** Store large or sensitive Task outputs.
*   **Meta Memory:** Enhance context with additional metadata.

### Drivers

Facilitate interaction with external resources and services, allowing for easy swapping of functionality.

#### LLM & Orchestration

*   Prompt Drivers
*   Assistant Drivers
*   Ruleset Drivers
*   Conversation Memory Drivers
*   Event Listener Drivers
*   Structure Run Drivers

#### Retrieval & Storage

*   Embedding Drivers
*   Rerank Drivers
*   Vector Store Drivers
*   File Manager Drivers
*   SQL Drivers

#### Multimodal

*   Image Generation Drivers
*   Text to Speech Drivers
*   Audio Transcription Drivers

#### Web

*   Web Search Drivers
*   Web Scraper Drivers

#### Observability

*   Observability Drivers

### Tools

Enable LLMs to interact with data and services. Explore [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/) and [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### Engines

Provide use-case-specific functionality by wrapping Drivers.

*   RAG Engine
*   Extraction Engine
*   Summary Engine
*   Eval Engine

### Additional Components

*   Rulesets
*   Loaders
*   Artifacts
*   Chunkers
*   Tokenizers

## Documentation

Find detailed information on installation, usage, and more in the official [docs](https://docs.griptape.ai/).

Also, check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Code Examples

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

## Versioning

Griptape uses [Semantic Versioning](https://semver.org/).

## Contributing

Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before contributing.

## License

Griptape is available under the Apache 2.0 License.