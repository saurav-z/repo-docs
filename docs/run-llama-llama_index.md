# LlamaIndex: Your Data Framework for LLM Applications 🦙

**Supercharge your Large Language Models (LLMs) with your own data using LlamaIndex, the open-source data framework.** ([View on GitHub](https://github.com/run-llama/llama_index))

LlamaIndex (formerly GPT Index) empowers you to build powerful LLM applications by providing the tools to ingest, structure, and access your data.

## Key Features

*   **Data Connectors:** Seamlessly ingest data from diverse sources like APIs, PDFs, documents, SQL databases, and more.
*   **Data Structuring:** Organize your data with efficient indices and graphs for optimal LLM performance.
*   **Advanced Retrieval & Querying:**  Retrieve relevant context and generate knowledge-augmented outputs from your LLM prompts.
*   **Flexible Integrations:**  Easily integrate with popular frameworks like LangChain, Flask, Docker, and ChatGPT.
*   **Modular Architecture:**  Customize and extend every component, from data connectors to query engines, to suit your specific needs.
*   **LlamaHub:** Access a vast library of community-contributed data loaders through [LlamaHub](https://llamahub.ai).

## Getting Started

LlamaIndex offers two primary installation paths:

*   **Starter Package:** `llama-index` - A comprehensive package including core LlamaIndex and a selection of integrations.
*   **Customized Package:** `llama-index-core` - Install the core library and add integrations from [LlamaHub](https://llamahub.ai/) for a tailored experience.

```bash
# Install the starter package
pip install llama-index
```

## Core Concepts

*   `llama_index.core`:  The core functionalities of LlamaIndex.
*   `llama_index.*`: Integration packages that work seamlessly with the core.

## Ecosystem

*   **LlamaHub:**  [https://llamahub.ai](https://llamahub.ai) - Community-driven data loaders.
*   **LlamaLab:** [https://github.com/run-llama/llama-lab](https://github.com/run-llama/llama-lab) - Explore cutting-edge AGI projects.

## Documentation

Comprehensive documentation is available at [https://docs.llamaindex.ai/en/stable/](https://docs.llamaindex.ai/en/stable/) for detailed tutorials, guides, and references.

## Example Usage

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Load documents from a directory
documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()

# Build an index
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Query your data
response = query_engine.query("YOUR_QUESTION")
print(response)
```

## Contributing

Contribute to LlamaIndex!  See the [Contribution Guide](CONTRIBUTING.md) for details on how to get involved.

## Dependencies

Project dependencies are managed with Poetry and can be found in the `pyproject.toml` file within each package folder.

## Citation

```
@software{Liu_LlamaIndex_2022,
author = {Liu, Jerry},
doi = {10.5281/zenodo.1234},
month = {11},
title = {{LlamaIndex}},
url = {https://github.com/jerryjliu/llama_index},
year = {2022}
}