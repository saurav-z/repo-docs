![Griptape](https://assets-global.website-files.com/65d658559223871198e78bca/65fb8d85c1ab3c9b858ab18a_Griptape%20logo%20dark.svg)

[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/griptape-ai/griptape/graph/badge.svg?token=HUBqUpl3NB)](https://codecov.io/github/griptape-ai/griptape)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/griptape)

# Griptape: Build Powerful GenAI Applications with Ease

Griptape is a Python framework designed to simplify the development of generative AI (genAI) applications, offering a streamlined approach to building and deploying AI-powered solutions. ([Original Repo](https://github.com/griptape-ai/griptape))

## Key Features

*   **Modular Architecture:** Build applications using core components like Agents, Pipelines, and Workflows.
*   **Flexible Structures:** Organize tasks sequentially or in parallel with Agents, Pipelines, and Workflows.
*   **Task Management:** Core building blocks for interacting with Engines, Tools, and other components.
*   **Memory Mechanisms:** Implement conversational memory, task memory, and meta memory for enhanced LLM interactions.
*   **Extensive Driver Support:** Interact with a variety of external resources and services, including LLMs, storage, and web services.
*   **Built-in & Custom Tools:** Leverage pre-built tools and create your own to empower LLMs with data and service interactions.
*   **Specialized Engines:** Utilize engines like RAG, Extraction, Summary, and Eval to address common use cases.
*   **Additional Components:** Employ Rulesets, Loaders, Artifacts, and Chunkers to refine LLM behavior and data handling.

## Core Components Explained

### Structures
*   **Agents**: Designed for Agent-specific behavior.
*   **Pipelines**: Organize a sequence of Tasks.
*   **Workflows**: Configure Tasks to operate in parallel.

### Tasks
Tasks are the core building blocks within Structures, enabling interaction with Engines, Tools, and other Griptape components.

### Memory
*   **Conversation Memory** enables LLMs to retain and retrieve information across interactions.
*   **Task Memory** keeps large or sensitive Task outputs off the prompt that is sent to the LLM.
*   **Meta Memory** enables passing in additional metadata to the LLM, enhancing the context and relevance of the interaction.

### Drivers
Drivers facilitate interactions with external resources and services in Griptape. 
They allow you to swap out functionality and providers with minimal changes to your business logic.

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

### Tools

Tools provide capabilities for LLMs to interact with data and services.
Griptape includes a variety of [built-in Tools](https://docs.griptape.ai/stable/griptape-framework/tools/official-tools/), and makes it easy to create [custom Tools](https://docs.griptape.ai/stable/griptape-framework/tools/custom-tools/).

### Engines

Engines wrap Drivers and provide use-case-specific functionality:

-   📊 **RAG Engine** is an abstraction for implementing modular Retrieval Augmented Generation (RAG) pipelines.
-   🛠️ **Extraction Engine** extracts JSON or CSV data from unstructured text.
-   📝 **Summary Engine** generates summaries from textual content.
-   ✅ **Eval Engine** evaluates and scores the quality of generated text.

### Additional Components

-   📐 **Rulesets** steer LLM behavior with minimal prompt engineering.
-   🔄 **Loaders** load data from various sources.
-   🏺 **Artifacts** allow for passing data of different types between Griptape components.
-   ✂️ **Chunkers** segment texts into manageable pieces for diverse text types.
-   🔢 **Tokenizers** count the number of tokens in a text to not exceed LLM token limits.

## Getting Started

Explore the [documentation](https://docs.griptape.ai/) for installation and usage guides.

## Examples

See the original README for the "Hello World" and "Task and Workflow" examples.

## Additional Resources

Check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Versioning

Griptape uses [Semantic Versioning](https://semver.org/).

## Contributing

Please review our [Contributing Guidelines](https://github.com/griptape-ai/griptape/blob/main/CONTRIBUTING.md) before contributing.

## License

Griptape is available under the Apache 2.0 License.