[![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)](https://github.com/griptape-ai/griptape)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

## Griptape: Build Powerful GenAI Applications with Ease

Griptape is a Python framework designed to simplify the development of generative AI applications. It offers a comprehensive suite of tools and abstractions for building sophisticated GenAI solutions.

**[Visit the Original Repo](https://github.com/griptape-ai/griptape)**

### Key Features

*   **Modular Architecture:** Build applications with Agents, Pipelines, and Workflows for structured GenAI operations.
*   **Versatile Task Management:** Utilize Tasks as core building blocks for interacting with Engines, Tools, and other Griptape components.
*   **Robust Memory Systems:** Leverage Conversation Memory, Task Memory, and Meta Memory to enhance context and interaction.
*   **Flexible Driver Support:** Interact with external resources and services using Drivers (LLMs, Retrieval, Storage, and more).
*   **Extensible Tool Ecosystem:** Use built-in Tools and create custom Tools for data interaction and service integration.
*   **Specialized Engines:** Utilize RAG Engine, Extraction Engine, Summary Engine, and Eval Engine for specific GenAI tasks.

### Core Components

*   **Structures:** Organize tasks:
    *   🤖 **Agents**: Single-task, agent-specific behavior.
    *   🔄 **Pipelines**: Sequential task execution.
    *   🌐 **Workflows**: Parallel task execution.
*   **Tasks:** Core building blocks enabling interactions.
*   **Memory:** Enhance context:
    *   💬 **Conversation Memory** for retaining information.
    *   🗃️ **Task Memory** to keep outputs off prompts.
    *   📊 **Meta Memory** to pass in additional metadata.
*   **Drivers:** Facilitate external interactions:
    *   🗣️ **Prompt Drivers**: LLM interaction.
    *   🤖 **Assistant Drivers**: "Assistant" services.
    *   📜 **Ruleset Drivers**: Ruleset application.
    *   🧠 **Conversation Memory Drivers**: Conversational data storage.
    *   📡 **Event Listener Drivers**: Framework events.
    *   🏗️ **Structure Run Drivers**: Cloud/local structure execution.
    *   🔢 **Embedding Drivers**: Vector embeddings.
    *   🔀 **Rerank Drivers**: Search result reranking.
    *   💾 **Vector Store Drivers**: Embedding management.
    *   🗂️ **File Manager Drivers**: File operations.
    *   🎨 **Image Generation Drivers**: Image creation.
    *   🗣️ **Text to Speech Drivers**: Text to speech.
    *   🎙️ **Audio Transcription Drivers**: Audio to text.
    *   🔍 **Web Search Drivers**: Web search.
    *   🌐 **Web Scraper Drivers**: Web data extraction.
    *   📈 **Observability Drivers**: Observability platform integration.
*   **Tools:** Enable data and service interaction.
*   **Engines:** Provide use-case specific functionality:
    *   📊 **RAG Engine**: Retrieval Augmented Generation.
    *   🛠️ **Extraction Engine**: Extracting structured data.
    *   📝 **Summary Engine**: Generating summaries.
    *   ✅ **Eval Engine**: Evaluating generated text.
*   **Additional Components:**
    *   📐 **Rulesets**: Steer LLM behavior.
    *   🔄 **Loaders**: Load data from sources.
    *   🏺 **Artifacts**: Pass data between components.
    *   ✂️ **Chunkers**: Text segmentation.
    *   🔢 **Tokenizers**: Token counting.

### Documentation

Detailed information on installation, usage, and examples can be found in the [official documentation](https://docs.griptape.ai/).

Explore free online courses at [Griptape Trade School](https://learn.griptape.ai/).

### Examples

**Hello World Example**

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

**Task and Workflow Example**

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

### Versioning

Griptape uses [Semantic Versioning](https://semver.org/).

### Contributing

Contributions are welcome! Please review the [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before submitting pull requests.

### License

Griptape is licensed under the Apache 2.0 License.