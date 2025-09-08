![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful GenAI Applications with Ease

**Griptape is a Python framework empowering developers to create robust and scalable generative AI applications.**  This framework simplifies the development of Generative AI (GenAI) applications with flexible abstractions for Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and more.  [Explore the Griptape Repository](https://github.com/griptape-ai/griptape).

## Key Features

*   **Modular Architecture:** Build GenAI applications with Agents, Pipelines, and Workflows for flexible task organization.
*   **Versatile Tasks:** Leverage core building blocks for interacting with Engines, Tools, and other Griptape components.
*   **Advanced Memory Management:** Utilize Conversation, Task, and Meta Memory to enhance LLM interactions.
*   **Extensive Driver Support:** Connect to external services with LLM, Retrieval, Storage, and Multimodal drivers.
*   **Pre-built Engines:** Implement use-case-specific functionality such as RAG, Extraction, Summary and Evaluation.
*   **Customizable Tools:** Integrate [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/) and easily create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/) to extend LLM capabilities.
*   **Additional Components:** Incorporate Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers for enhanced control.

## Core Components

### Structures

*   **Agents:** Task-specific, configured for unique behavior.
*   **Pipelines:** Orchestrate sequences of tasks for efficient workflows.
*   **Workflows:** Enable parallel task execution.

### Tasks

*   The fundamental building blocks for interaction with Engines, Tools, and other components.

### Memory

*   **Conversation Memory:** Maintain context across interactions.
*   **Task Memory:** Manage large outputs and sensitive data.
*   **Meta Memory:** Enrich context with additional metadata.

### Drivers

*   Facilitate interactions with external resources and services.

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

*   Enhance LLM capabilities by connecting them with data and services.

### Engines

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

Comprehensive documentation is available at [https://docs.griptape.ai/](https://docs.griptape.ai/) to guide you through installation and usage.

## Examples

### Hello World

A minimal example to get you started:

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

A concise example using griptape to research open source projects:

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

Griptape adheres to [Semantic Versioning](https://semver.org/).

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before submitting any pull requests.

## License

Griptape is licensed under the Apache 2.0 License.