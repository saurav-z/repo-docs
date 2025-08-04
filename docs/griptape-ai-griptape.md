![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful AI Applications with Ease

**Griptape is a Python framework that simplifies the development of generative AI (genAI) applications, offering flexible abstractions for LLMs, Retrieval-Augmented Generation (RAG), and more.**

## Key Features

*   **Modular Structure:** Build AI applications with Agents, Pipelines, and Workflows for organized task execution.
*   **Versatile Task Management:** Utilize Tasks as the core building blocks, interacting with Engines, Tools, and other components.
*   **Intelligent Memory:** Leverage Conversation, Task, and Meta Memory to enhance LLM context and improve interactions.
*   **Extensive Driver Support:** Integrate with external resources and services with LLM & Orchestration, Retrieval & Storage, Multimodal, Web, and Observability drivers.
*   **Robust Tooling:** Access a wide range of built-in Tools and easily create custom tools to empower your AI applications.
*   **Specialized Engines:** Utilize engines like RAG Engine, Extraction Engine, Summary Engine, and Eval Engine for specialized use cases.
*   **Additional Components:** Employ Rulesets, Loaders, Artifacts, and Chunkers to refine and streamline your AI workflows.

## Core Components

### 🏗️ Structures

*   🤖 **Agents:** Single task, agent-specific behavior.
*   🔄 **Pipelines:** Sequenced Tasks for output flow.
*   🌐 **Workflows:** Parallel Task execution.

### 📝 Tasks

Tasks are the core building blocks within Structures, enabling interaction with Engines, Tools, and other Griptape components.

### 🧠 Memory

*   💬 **Conversation Memory:** Retain information across interactions.
*   🗃️ **Task Memory:** Keeps large or sensitive outputs off the prompt.
*   📊 **Meta Memory:** Pass additional metadata to LLM.

### 🚗 Drivers

Drivers facilitate interactions with external resources and services in Griptape.

#### LLM & Orchestration

*   🗣️ **Prompt Drivers:** Manage textual and image interactions with LLMs.
*   🤖 **Assistant Drivers:** Enable interactions with various “assistant” services.
*   📜 **Ruleset Drivers:** Load and apply rulesets from external sources.
*   🧠 **Conversation Memory Drivers:** Store and retrieve conversational data.
*   📡 **Event Listener Drivers:** Forward framework events to external services.
*   🏗️ **Structure Run Drivers:** Execute structures locally or in the cloud.

#### Retrieval & Storage

*   🔢 **Embedding Drivers:** Generate vector embeddings from textual inputs.
*   🔀 **Rerank Drivers:** Rerank search results for improved relevance.
*   💾 **Vector Store Drivers:** Manage the storage and retrieval of embeddings.
*   🗂️ **File Manager Drivers:** Handle file operations on local and remote storage.
*   💼 **SQL Drivers:** Interact with SQL databases.

#### Multimodal

*   🎨 **Image Generation Drivers:** Create images from text descriptions.
*   🗣️ **Text to Speech Drivers:** Convert text to speech.
*   🎙️ **Audio Transcription Drivers:** Convert audio to text.

#### Web

*   🔍 **Web Search Drivers:** Search the web for information.
*   🌐 **Web Scraper Drivers:** Extract data from web pages.

#### Observability

*   📈 **Observability Drivers:** Send trace and event data to observability platforms.

### 🔧 Tools

Tools provide capabilities for LLMs to interact with data and services.
Griptape includes a variety of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/), and makes it easy to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### 🚂 Engines

Engines wrap Drivers and provide use-case-specific functionality:

*   📊 **RAG Engine:** Implement modular Retrieval Augmented Generation (RAG) pipelines.
*   🛠️ **Extraction Engine:** Extract JSON or CSV data from unstructured text.
*   📝 **Summary Engine:** Generate summaries from textual content.
*   ✅ **Eval Engine:** Evaluate and score generated text.

### 📦 Additional Components

*   📐 **Rulesets:** Steer LLM behavior with minimal prompt engineering.
*   🔄 **Loaders:** Load data from various sources.
*   🏺 **Artifacts:** Allow for passing data of different types between Griptape components.
*   ✂️ **Chunkers:** Segment texts into manageable pieces for diverse text types.
*   🔢 **Tokenizers:** Count the number of tokens in a text to not exceed LLM token limits.

## Getting Started

Explore the [documentation](https://docs.griptape.ai/) for detailed installation and usage instructions.

Check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

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

Output:

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

Interested in contributing?  Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md).

## License

Griptape is available under the Apache 2.0 License.

[Back to top](#griptape-build-powerful-ai-applications-with-ease)
```
Key improvements and SEO optimizations:

*   **Clear Headline:**  Uses a strong, keyword-rich headline.
*   **Concise Hook:**  A one-sentence summary as the first line.
*   **Keyword Optimization:**  Includes relevant keywords like "AI," "genAI," "LLMs," "RAG," and "Python framework."
*   **Structured Content:** Uses clear headings and bullet points for readability.
*   **Comprehensive Feature List:** Highlights key advantages in a user-friendly way.
*   **Internal Linking:** Provides links to different parts of the README.
*   **Call to Action:** Encourages exploration of documentation and online courses.
*   **Contextual Examples:** Includes the original code examples.
*   **Clear Structure and Formatting:** Improved formatting for better readability.
*   **Back to Top Link:** Added a link to quickly jump back to the top of the README.
*   **Emphasis on Benefits:** Highlights what users can *do* with Griptape.