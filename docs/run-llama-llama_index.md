# 🗂️ LlamaIndex: Your Gateway to LLM-Powered Applications

**Unlock the power of your data with LlamaIndex, a versatile framework designed to connect your data with Large Language Models, enabling you to build intelligent and context-aware applications.** ([See Original Repo](https://github.com/run-llama/llama_index))

[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-index)](https://pypi.org/project/llama-index/)
[![Build](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml/badge.svg)](https://github.com/run-llama/llama_index/actions/workflows/build_package.yml)
[![GitHub contributors](https://img.shields.io/github/contributors/jerryjliu/llama_index)](https://github.com/jerryjliu/llama_index/graphs/contributors)
[![Discord](https://img.shields.io/discord/1059199217496772688)](https://discord.gg/dGcwcsnxhU)
[![Twitter](https://img.shields.io/twitter/follow/llama_index)](https://x.com/llama_index)
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)
[![Ask AI](https://img.shields.io/badge/Phorm-Ask_AI-%23F2777A.svg?&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNSIgaGVpZ2h0PSI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik00LjQzIDEuODgyYTEuNDQgMS40NCAwIDAgMS0uMDk4LjQyNmMtLjA1LjEyMy0uMTE1LjIzLS4xOTIuMzIyLS4wNzUuMDktLjE2LjE2NS0uMjU1LjIyNmExLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxMmMtLjA5OS4wMTItLjE5Mi4wMTQtLjI3OS4wMDZsLTEuNTkzLS4xNHYtLjQwNmgxLjY1OGMuMDkuMDAxLjE3LS4xNjkuMjQ2LS4xOTFhLjYwMy42MDMgMCAwIDAgLjItLjEwNi41MjkuNTI5IDAgMCAwIC4xMzgtLjE3LjY1NC42NTQgMCAwIDAgLjA2NS0uMjRsLjAyOC0uMzJhLjkzLjkzIDAgMCAwLS4wMzYtLjI0OS41NjcuNTY3IDAgMCAwLS4xMDMtLjIuNTAyLjUwMiAwIDAgMC0uMTY4LS4xMzguNjA4LjYwOCAwIDAgMC0uMjQtLjA2N0wyLjQzNy43MjkgMS42MjUuNjcxYS4zMjIuMzIyIDAgMCAwLS4yMzIuMDU4LjM3NS4zNzUgMCAwIDAtLjExNi4yMzJsLS4xMTYgMS40NS0uMDU4LjY5Ny0uMDU4Ljc1NEwuNzA1IDRsLS4zNTctLjA3OUwuNjAyLjkwNkMuNjE3LjcyNi42NjMuNTc0LjczOS40NTRhLjk1OC45NTggMCAwIDEgLjI3NC0uMjg1Ljk3MS45NzEgMCAwIDEgLjMzNy0uMTRjLjExOS0uMDI2LjIyNy0uMDM0LjMyNS0uMDI2TDMuMjMyLjE2Yy4xNTkuMDE0LjMzNi4wMy40NTkuMDgyYTEuMTczIDEuMTczIDAgMCAxIC41NDUuNDQ3Yy4wNi4wOTQuMTA5LjE5Mi4xNDQuMjkzYTEuMzkyIDEuMzkyIDAgMCAxIC4wNzguNThsLS4wMjkuMzJaIiBmaWxsPSIjRjI3NzdBIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=)](https://www.phorm.ai/query?projectId=c5863b56-6703-4a5d-87b6-7e6031bf16b6)

## Key Features:

*   **Data Connectors:** Seamlessly ingest data from diverse sources like APIs, PDFs, documents, and SQL databases.
*   **Data Structuring:** Organize your data using indices and graphs for efficient LLM interaction.
*   **Advanced Retrieval/Query Interface:** Easily retrieve relevant context and generate knowledge-augmented outputs with LLM input prompts.
*   **Integration Flexibility:**  Integrate effortlessly with popular frameworks like LangChain, Flask, Docker, and ChatGPT.
*   **Modular Design:** Choose from core LlamaIndex or customize with a wide selection of integrations for LLMs, embedding, and vector store providers.

## Getting Started

LlamaIndex offers two main ways to get started:

1.  **Starter Package:** Use the `llama-index` Python package for a quick start with core LlamaIndex and a selection of integrations.
2.  **Customized Installation:** Install `llama-index-core` and add your desired integrations from [LlamaHub](https://llamahub.ai/), which features over 300 integration packages.

## Core Concepts

*   **Core Package (`llama-index-core`):** The foundational package with essential components.
*   **Integrations:**  Specialized packages for specific LLMs, embedding models, and vector stores.
*   **Namespace:** Import statements containing `core` indicate usage of the core package, while those without refer to integrations.

```python
# Example:  Importing from core and an integration
from llama_index.core.llms import LLM  # Core
from llama_index.llms.openai import OpenAI # Integration
```

## Important Links:

*   **LlamaIndex.TS (Typescript/Javascript):** [https://github.com/run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS)
*   **Documentation:** [https://docs.llamaindex.ai/en/stable/](https://docs.llamaindex.ai/en/stable/)
*   **X (formerly Twitter):** [https://x.com/llama_index](https://x.com/llama_index)
*   **LinkedIn:** [https://www.linkedin.com/company/llamaindex/](https://www.linkedin.com/company/llamaindex/)
*   **Reddit:** [https://www.reddit.com/r/LlamaIndex/](https://www.reddit.com/r/LlamaIndex/)
*   **Discord:** [https://discord.gg/dGcwcsnxhU](https://discord.gg/dGcwcsnxhU)

## Ecosystem:

*   **LlamaHub:** [https://llamahub.ai](https://llamahub.ai) (Community library of data loaders)
*   **LlamaLab:** [https://github.com/run-llama/llama-lab](https://github.com/run-llama/llama-lab) (Cutting-edge AGI projects)

## 🚀 Overview

**Note:**  For the latest updates and detailed information, please refer to the comprehensive documentation.

### Context

Large Language Models (LLMs) excel at knowledge generation and reasoning but are limited by their pre-training data.

### Solution

LlamaIndex empowers you to augment LLMs with your private data through a comprehensive toolkit.

### Key Benefits:

*   **Data Ingestion:** Connect to diverse data sources (APIs, PDFs, etc.).
*   **Data Structuring:** Structure data for optimal LLM use.
*   **Advanced Querying:** Retrieve context and generate knowledge-augmented outputs.
*   **Flexible Integration:** Integrate seamlessly with your existing application framework.
*   **Ease of Use:** Easy-to-use APIs for beginners and lower-level APIs for customization.

## 💡 Contributing

Contribute to the LlamaIndex ecosystem by contributing to core features or building integrations.  See the [Contribution Guide](CONTRIBUTING.md) for more details.

## 📄 Documentation

Access detailed documentation, tutorials, and guides [here](https://docs.llamaindex.ai/en/stable/).

## 💻 Example Usage

```bash
# Install necessary packages
pip install llama-index-core
pip install llama-index-llms-openai
pip install llama-index-llms-replicate
pip install llama-index-embeddings-huggingface
```

```python
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load Data
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()

# Build Index
index = VectorStoreIndex.from_documents(documents)

#  Query
query_engine = index.as_query_engine()
query_engine.query("YOUR_QUESTION")

# Persist to disk
index.storage_context.persist()
```

**Example using Replicate (non-OpenAI LLM):**

```python
import os
os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_TOKEN"

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

# Set up LLM, tokenizer, and embed model
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(model=llama2_7b_chat, temperature=0.01, additional_kwargs={"top_p": 1, "max_new_tokens": 300})
Settings.tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load data
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(documents)
```

To reload the index:
```python
from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)
```

## 🔧 Dependencies

Dependencies are managed using poetry, consult the `pyproject.toml` file in each package folder.

```bash
cd <desired-package-folder>
pip install poetry
poetry install --with dev
```

## 📖 Citation

If you use LlamaIndex in your research, please cite:

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