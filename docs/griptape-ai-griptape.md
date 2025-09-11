![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful GenAI Applications with Ease

**Griptape is a Python framework that simplifies the development of cutting-edge Generative AI applications, offering flexible abstractions for LLMs, RAG, and more.** Explore the power of Griptape and streamline your AI development workflow.

## Key Features:

*   **Modular Architecture:** Build and organize AI applications using Agents, Pipelines, and Workflows.
*   **Flexible Memory:** Leverage Conversation, Task, and Meta Memory to enhance context and improve interactions.
*   **Extensive Driver Support:** Interact with various LLMs, retrieval systems, storage solutions, and multimodal services.
*   **Built-in and Customizable Tools:** Integrate with data and services using a variety of tools.
*   **Powerful Engines:** Use RAG, Extraction, Summary, and Eval engines to build advanced AI applications.
*   **Additional Components:** Utilize Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers to customize your AI applications.

## Core Components

### Structures

*   **Agents:** Focused on a single Task for Agent-specific behavior.
*   **Pipelines:** Orchestrate a sequence of Tasks.
*   **Workflows:** Configure Tasks for parallel execution.

### Tasks

Tasks are the core building blocks, enabling interaction with Engines, Tools, and other Griptape components.

### Memory

*   **Conversation Memory:** Retain and retrieve information across interactions.
*   **Task Memory:** Keep large or sensitive Task outputs off the prompt.
*   **Meta Memory:** Enhance context with additional metadata.

### Drivers

Drivers facilitate interactions with external resources and services.

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

Tools provide capabilities for LLMs to interact with data and services.  Griptape includes a variety of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/), and makes it easy to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### Engines

Engines wrap Drivers and provide use-case-specific functionality:

*   **RAG Engine**
*   **Extraction Engine**
*   **Summary Engine**
*   **Eval Engine**

### Additional Components

*   **Rulesets**
*   **Loaders**
*   **Artifacts**
*   **Chunkers**
*   **Tokenizers**

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

## Documentation

Find detailed information on installation and usage at the [Griptape Docs](https://docs.griptape.ai/).

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before getting started.

## License

Griptape is released under the Apache 2.0 License.

## Get Started

Explore the [Griptape Repository](https://github.com/griptape-ai/griptape) to start building your next-generation AI applications today!