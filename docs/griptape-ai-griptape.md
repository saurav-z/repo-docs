![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

## Griptape: Build Powerful GenAI Applications with Ease

Griptape is a Python framework that streamlines the development of Generative AI applications, providing robust tools for LLMs, Retrieval-Augmented Generation (RAG), and more.  [Explore the Griptape repository](https://github.com/griptape-ai/griptape) to get started!

### Key Features

*   **Modular Structures:**
    *   🤖 **Agents:** Single-task configurations for agent-specific behaviors.
    *   🔄 **Pipelines:** Organize tasks sequentially.
    *   🌐 **Workflows:** Enable parallel task execution.
*   **Versatile Tasks:** Core building blocks for interacting with Engines, Tools, and other Griptape components.
*   **Intelligent Memory:**
    *   💬 **Conversation Memory:** Retain context across interactions.
    *   🗃️ **Task Memory:** Manage large outputs efficiently.
    *   📊 **Meta Memory:** Enhance context with additional metadata.
*   **Extensible Drivers:** Facilitate interactions with external services, including:
    *   🗣️ **Prompt Drivers:** Manage LLM interactions.
    *   🤖 **Assistant Drivers:** Integrate with assistant services.
    *   🧠 **Conversation Memory Drivers:** Store and retrieve conversational data.
    *   🌐 **Web Search Drivers:** Search the web for information.
    *   💾 **Vector Store Drivers:** Manage embeddings.
    *   🎨 **Image Generation Drivers:** Create images from text.
    *   And more for RAG, SQL, and other integrations.
*   **Powerful Tools:**  A range of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/) and custom tool creation capabilities.
*   **Specialized Engines:** Use-case specific functionality with:
    *   📊 **RAG Engine**: Modular Retrieval Augmented Generation.
    *   🛠️ **Extraction Engine**: Extract data from unstructured text.
    *   📝 **Summary Engine**: Generate summaries.
    *   ✅ **Eval Engine**: Evaluate generated text quality.
*   **Additional Components:** Rulesets, Loaders, Artifacts, Chunkers, and Tokenizers.

### Getting Started

*   **Documentation:**  Comprehensive [documentation](https://docs.griptape.ai/) for installation and usage.
*   **Learning Resources:** Free online courses are available via [Griptape Trade School](https://learn.griptape.ai/).

### Example Code

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

### Advanced Example

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

### Versioning

Uses [Semantic Versioning](https://semver.org/).

### Contributing

Review the [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before contributing.

### License

Griptape is available under the Apache 2.0 License.