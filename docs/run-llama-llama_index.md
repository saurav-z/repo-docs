# LlamaIndex: Build LLM Applications with Your Data

**LlamaIndex empowers you to seamlessly connect your data with Large Language Models (LLMs), enabling powerful and knowledge-rich applications.**

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index)](https://pypi.org/project/llama-index/)
[![Build](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml/badge.svg)](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml)
[![GitHub contributors](https://img.shields.io/github/contributors/jerryjliu/llama_index)](https://github.com/jerryjliu/llama_index/graphs/contributors)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)
[![Ask AI](https://img.shields.io/badge/Phorm-Ask_AI-%23F2777A.svg?&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNSIgaGVpZ2h0PSI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik00LjQzIDEuODgyYTEuNDQgMS40NCAwIDAgMS0uMDk4LjQyNmMtLjA1LjEyMy0uMTE1LjIzLS4xOTIuMzIyLS4wNzUuMDktLjE2LjE2NS0uMjU1LjIyNmExLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxMmMtLjA5OS4wMTItLjE5Mi4wMTQtLjI3OS4wMDZsLTEuNTkzLS4xNHYtLjQwNmgxLjY1OGMuMDkuMDAxLjE3LS4xNjkuMjQ2LS4xOTFhLjYwMy42MDMgMCAwIDAgLjItLjEwNi41MjkuNTI5IDAgMCAwIC4xMzgtLjE3LjY1NC42NTQgMCAwIDAgLjA2NS0uMjRsLjAyOC0uMzJhLjkzLjkzIDAgMCAwLS4wMzYtLjI0OS41NjcuNTY3IDAgMCAwLS4xMDMtLjIuNTAyLjUwMiAwIDAgMC0uMTY4LS4xMzguNjA4LjYwOCAwIDAgMC0uMjQtLjA2N0wyLjQzNy43MjkgMS42MjUuNjcxYS4zMjIuMzIyIDAgMCAwLS4yMzIuMDU4LjM3NS4zNzUgMCAwIDAtLjExNi4yMzJsLS4xMTYgMS40NS0uMDU4LjY5Ny0uMDU4Ljc1NEwuNzA1IDRsLS4zNTctLjA3OUwuNjAyLjkwNkMuNjE3LjcyNi42NjMuNTc0LjczOS40NTRhLjk1OC45NTggMCAwIDEgLjI3NC0uMjg1Ljk3MS45NzEgMCAwIDEgLjMzNy0uMTRjLjExOS0uMDI2LjIyNy0uMDM0LjMyNS0uMDI2TDMuMjMyLjE2Yy4xNTkuMDE0LjMzNi4wMy40NTkuMDgyYTEuMTczIDEuMTczIDAgMCAxIC41NDUuNDQ3Yy4wNi4wOTQuMTA5LjE5Mi4xNDQuMjkzYTEuMzkyIDEuMzkyIDAgMCAxIC4wNzguNThsLS4wMjkuMzJaIiBmaWxsPSIjRjI3NzdBIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=)](https://www.phorm.ai/query?projectId=c5863b56-6703-4a5d-87b6-7e6031bf16b6)

**[Visit the LlamaIndex GitHub Repository](https://github.com/run-llama/llama_index)**

LlamaIndex (formerly GPT Index) is a powerful data framework designed to streamline the development of LLM-powered applications. It provides the tools and infrastructure necessary to integrate your own data with LLMs, enabling knowledge generation and reasoning capabilities.

**Key Features:**

*   **Data Connectors:** Ingest data from diverse sources like APIs, PDFs, documents, and SQL databases.
*   **Data Structuring:** Organize data using efficient structures (indices, graphs) optimized for LLM interaction.
*   **Advanced Retrieval/Query Interface:**  Retrieve relevant context and generate knowledge-augmented output based on LLM input prompts.
*   **Seamless Integration:** Easily integrates with popular frameworks like LangChain, Flask, Docker, and more.
*   **User-Friendly API:** Offers a high-level API for quick data ingestion and querying, alongside lower-level APIs for customization.
*   **Extensive Ecosystem:** Integrates with a vast array of LLMs, embeddings, and vector stores via [LlamaHub](https://llamahub.ai/).

### Important Links

*   **LlamaIndex.TS (Typescript/Javascript):** [https://github.com/run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS)
*   **Documentation:** [https://docs.llamaindex.ai/en/stable/](https://docs.llamaindex.ai/en/stable/)
*   **X (formerly Twitter):** [https://x.com/llama_index](https://x.com/llama_index)
*   **LinkedIn:** [https://www.linkedin.com/company/llamaindex/](https://www.linkedin.com/company/llamaindex/)
*   **Reddit:** [https://www.reddit.com/r/LlamaIndex/](https://www.reddit.com/r/LlamaIndex/)
*   **Discord:** [https://discord.gg/dGcwcsnxhU](https://discord.gg/dGcwcsnxhU)

### Ecosystem

*   **LlamaHub (Community Library of Data Loaders):** [https://llamahub.ai](https://llamahub.ai)
*   **LlamaLab (Cutting-Edge AGI Projects):** [https://github.com/run-llama/llama-lab](https://github.com/run-llama/llama-lab)

## 🚀 Overview

**Note:** For the most up-to-date information, please refer to the official documentation linked above.

### Context

Large Language Models excel at knowledge generation and reasoning but are pre-trained on public data.

### The Challenge

How to best augment LLMs with our own private, custom data?

### Solution: LlamaIndex

LlamaIndex provides a comprehensive framework to help you build LLM applications.  It offers:

*   **Data Ingestion:**  Connectors to bring in various data formats (APIs, PDFs, Docs, SQL, etc.).
*   **Data Structuring:** Indexing and graph construction for efficient use with LLMs.
*   **Retrieval & Querying:** Advanced interfaces for retrieving context and generating informed responses.
*   **Application Framework Integration:**  Easy integration with popular frameworks.

## 💡 Contributing

Contributions are highly encouraged!  See the [Contribution Guide](CONTRIBUTING.md) for details. New integrations are welcome.

## 📄 Documentation

Comprehensive documentation, including tutorials and guides, is available [here](https://docs.llamaindex.ai/en/latest/).

## 💻 Example Usage

```bash
# Example: Install Core and Open AI support
pip install llama-index-core
pip install llama-index-llms-openai
```

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Load Data
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()

# Create Index
index = VectorStoreIndex.from_documents(documents)
```

```python
# Query the Index
query_engine = index.as_query_engine()
query_engine.query("YOUR_QUESTION")
```

## 🔧 Dependencies

LlamaIndex utilizes poetry for package management.  See `pyproject.toml` in each package folder for dependencies.

```bash
cd <desired-package-folder>
pip install poetry
poetry install --with dev
```

## 📖 Citation

```
@software{Liu_LlamaIndex_2022,
author = {Liu, Jerry},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{LlamaIndex}},
url = {https://github.com/jerryjliu/llama_index},
year = {2022}
}
```