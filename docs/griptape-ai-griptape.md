# Griptape: Build Powerful GenAI Applications with Ease

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

**Griptape is a Python framework that simplifies building and deploying generative AI applications, providing robust tools for LLMs, RAG, and more.**  

## Key Features

*   **Modular Architecture:** Build with Agents, Pipelines, and Workflows for flexible application design.
*   **Core Components:**
    *   **Agents**: Single-task structures for specific behaviors.
    *   **Pipelines**: Sequence tasks for data flow.
    *   **Workflows**: Execute tasks in parallel.
    *   **Tasks**: Foundation for interacting with Engines, Tools, and other components.
*   **Memory Management:**
    *   **Conversation Memory:** Enables LLMs to retain information.
    *   **Task Memory:** Keeps large or sensitive outputs off of the prompt.
    *   **Meta Memory:** Passes metadata to the LLM for context.
*   **Extensive Driver Support:**
    *   **Prompt Drivers:** LLM interaction (text & image).
    *   **Assistant Drivers:** Integrations with "assistant" services.
    *   **Ruleset Drivers:** Application of external rules.
    *   **Conversation Memory Drivers:** Data storage and retrieval.
    *   **Event Listener Drivers:** Forward framework events to external services.
    *   **Structure Run Drivers:** Run structures locally or in the cloud.
    *   **Embedding Drivers:** Generate vector embeddings.
    *   **Rerank Drivers:** Improve search relevance.
    *   **Vector Store Drivers:** Manage embeddings.
    *   **File Manager Drivers:** Handle file operations.
    *   **SQL Drivers:** Database interaction.
    *   **Image Generation Drivers:** Image creation.
    *   **Text-to-Speech Drivers:** Text conversion.
    *   **Audio Transcription Drivers:** Audio conversion.
    *   **Web Search Drivers:** Web information retrieval.
    *   **Web Scraper Drivers:** Data extraction from web pages.
    *   **Observability Drivers:** Send trace and event data.
*   **Tools Ecosystem:** Integrates with data and external services with built-in and custom tools.
*   **Powerful Engines:**
    *   **RAG Engine:** Implements Retrieval Augmented Generation (RAG).
    *   **Extraction Engine:** Extracts JSON or CSV.
    *   **Summary Engine:** Generates summaries.
    *   **Eval Engine:** Evaluates generated text.
*   **Additional Components:** Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers.

## Documentation

Explore detailed information on installation, usage, and more in the [Griptape documentation](https://docs.griptape.ai/).
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

## Versioning

Griptape follows [Semantic Versioning](https://semver.org/).

## Contributing

Contributions are welcome! See the [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) for more details.

## License

Griptape is available under the Apache 2.0 License.

[Back to the Griptape GitHub Repository](https://github.com/griptape-ai/griptape)