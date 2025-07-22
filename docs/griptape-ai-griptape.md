![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

## Griptape: Build Powerful Generative AI Applications with Ease

Griptape is a Python framework designed to streamline the development of sophisticated generative AI applications, offering flexible abstractions for LLMs, Retrieval-Augmented Generation (RAG), and much more.

**[Explore the Griptape Repository](https://github.com/griptape-ai/griptape)**

### Key Features

*   **Modular Architecture**: Build AI applications with core components like Agents, Pipelines, and Workflows for structured execution.
*   **Flexible Task Management**: Utilize Tasks as fundamental building blocks for interacting with Engines, Tools, and other Griptape components.
*   **Advanced Memory Capabilities**: Employ Conversation Memory, Task Memory, and Meta Memory to enhance context and improve LLM performance.
*   **Extensive Driver Support**: Integrate with diverse external resources and services using drivers for LLMs, Retrieval, Storage, Multimodal, Web, and Observability.
*   **Versatile Tool Ecosystem**: Leverage built-in tools and easily create custom tools to empower LLMs with data and service interactions.
*   **Specialized Engines**: Utilize RAG Engine, Extraction Engine, Summary Engine, and Eval Engine to meet specific use-case needs.
*   **Additional Components**: Utilize Rulesets, Loaders, Artifacts, and Chunkers to fine-tune LLM behavior and data handling.

### Core Components

Griptape's architecture comprises several key components:

*   **Structures**:
    *   **Agents**: Single-task configurations for agent-specific behavior.
    *   **Pipelines**: Sequential task execution.
    *   **Workflows**: Parallel task execution.
*   **Tasks**: Core units for interacting with Engines, Tools, and components.
*   **Memory**:
    *   **Conversation Memory**: Retain context across interactions.
    *   **Task Memory**: Manage large or sensitive task outputs.
    *   **Meta Memory**: Pass metadata for context enhancement.
*   **Drivers**:
    *   **LLM & Orchestration**: Prompt, Assistant, Ruleset, Conversation Memory, Event Listener, and Structure Run Drivers.
    *   **Retrieval & Storage**: Embedding, Rerank, Vector Store, File Manager, and SQL Drivers.
    *   **Multimodal**: Image Generation, Text to Speech, and Audio Transcription Drivers.
    *   **Web**: Web Search and Web Scraper Drivers.
    *   **Observability**: Observability Drivers.
*   **Tools**: Provide capabilities for LLMs to interact with data and services.
*   **Engines**:
    *   RAG Engine
    *   Extraction Engine
    *   Summary Engine
    *   Eval Engine
*   **Additional Components**: Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers.

### Documentation

Comprehensive documentation is available at [https://docs.griptape.ai/](https://docs.griptape.ai/).
Check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

### Examples

*   **Hello World:**

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
*   **Task and Workflow Example:**

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
    ```mermaid
        graph TD;
        griptape-->summary;
        langchain-->summary;
        pydantic-ai-->summary;
        crew-ai-->summary;
    ```

### Versioning

Griptape follows [Semantic Versioning](https://semver.org/).

### Contributing

Contributions are welcome! Please review the [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before contributing.

### License

Griptape is available under the Apache 2.0 License.