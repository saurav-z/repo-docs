![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful AI Applications with Ease

**Griptape is a Python framework that simplifies building generative AI applications by providing flexible abstractions for LLMs, RAG, and more.**  

Griptape offers a modular and intuitive approach to developing AI applications, allowing you to build complex systems with ease.

## Key Features

*   **Structures:** Organize your AI workflows with Agents, Pipelines, and Workflows.
*   **Tasks:**  Core building blocks for interacting with Engines, Tools, and other Griptape components.
*   **Memory:**
    *   Conversation Memory for retaining context across interactions.
    *   Task Memory to manage large outputs.
    *   Meta Memory for enriching prompts.
*   **Drivers:** Easily swap out LLMs, search providers, and other services with minimal code changes.
    *   **LLM & Orchestration Drivers:** Prompt, Assistant, Ruleset, Conversation Memory, Event Listener, and Structure Run Drivers.
    *   **Retrieval & Storage Drivers:** Embedding, Rerank, Vector Store, File Manager, and SQL Drivers.
    *   **Multimodal Drivers:** Image Generation, Text to Speech, and Audio Transcription Drivers.
    *   **Web Drivers:** Web Search and Web Scraper Drivers.
    *   **Observability Drivers:** Send trace and event data.
*   **Tools:**  Empower LLMs with the ability to interact with data and services.  Includes [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/) and easy [custom Tool](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/) creation.
*   **Engines:** Use-case-specific functionality built on top of Drivers: RAG Engine, Extraction Engine, Summary Engine, and Eval Engine.
*   **Additional Components:** Rulesets, Loaders, Artifacts, and Chunkers for flexible control and data management.

## Getting Started

Explore the comprehensive [documentation](https://docs.griptape.ai/) for installation and usage instructions.
Check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Example: Simple Kickflip with Griptape

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

## Example: Researching Open Source Projects with Workflows

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

Contributions are welcome! Please review the [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md).

## License

Griptape is available under the Apache 2.0 License.

[Back to Top](#griptape-build-powerful-ai-applications-with-ease)
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:** "Griptape: Build Powerful AI Applications with Ease" is a strong, keyword-rich title.
*   **One-Sentence Hook:** The introductory sentence immediately grabs attention and highlights the core benefit.
*   **Keyword Optimization:** The text is peppered with relevant keywords like "AI applications," "LLMs," "RAG," and "Python framework."
*   **Bulleted Key Features:**  Makes it easy for users to quickly scan and understand the core benefits.
*   **Organized Headings:** Improves readability and helps with SEO by structuring the content logically.
*   **Links:**  Includes links back to the original repo and the documentation, as requested.
*   **Actionable:** Provides clear instructions for getting started (documentation, examples).
*   **Visuals:** Maintains the logo, badges, and the Mermaid diagram for better engagement.
*   **Summarization:**  Condenses the original README while retaining essential information.
*   **Back to Top anchor** makes navigation easier.