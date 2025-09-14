<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: Supercharge Your LLMs with Simple and Fast Retrieval-Augmented Generation

<p align="center">
  <a href="https://github.com/HKUDS/LightRAG" target="_blank">
    <img src="https://img.shields.io/badge/🔥Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2410.05779" target="_blank">
    <img src="https://img.shields.io/badge/📄arXiv-2410.05779-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e" alt="arXiv">
  </a>
  <a href="https://github.com/HKUDS/LightRAG/stargazers" target="_blank">
    <img src="https://img.shields.io/github/stars/HKUDS/LightRAG?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e" alt="Stars">
  </a>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/🐍Python-3.10-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python">
    <a href="https://pypi.org/project/lightrag-hku/" target="_blank">
        <img src="https://img.shields.io/pypi/v/lightrag-hku.svg?style=for-the-badge&logo=pypi&logoColor=white&labelColor=1a1a2e&color=ff6b6b" alt="PyPI">
    </a>
</p>
<p align="center">
    <a href="https://discord.gg/yF2MmDJyGJ" target="_blank">
        <img src="https://img.shields.io/badge/💬Discord-Community-7289da?style=for-the-badge&logo=discord&logoColor=white&labelColor=1a1a2e" alt="Discord">
    </a>
    <a href="https://github.com/HKUDS/LightRAG/issues/285" target="_blank">
        <img src="https://img.shields.io/badge/💬WeChat-Group-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=1a1a2e" alt="WeChat">
    </a>
</p>
<p align="center">
    <a href="README-zh.md" target="_blank">
        <img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge" alt="Chinese Version">
    </a>
    <a href="README.md" target="_blank">
        <img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge" alt="English Version">
    </a>
</p>

</div>

<div align="center" style="margin: 30px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="800" alt="LightRAG Demo">
</div>

LightRAG simplifies and accelerates Retrieval-Augmented Generation (RAG), empowering you to build intelligent applications with ease.  **[Explore the LightRAG repository](https://github.com/HKUDS/LightRAG) to get started!**

---

## Key Features

*   **Fast and Efficient RAG:** Optimized for speed and performance, enabling quick retrieval and generation.
*   **Flexible Storage Options:** Supports various storage backends including JsonKVStorage, PGKVStorage, RedisKVStorage, MongoKVStorage, NetworkXStorage, Neo4JStorage, PGGraphStorage, etc.
*   **Multimodal Document Processing:** Integrates with [RAG-Anything](https://github.com/HKUDS/RAG-Anything) for seamless handling of PDFs, images, and more.
*   **Knowledge Graph Management:** Create, edit, and delete entities and relationships to build rich, interconnected knowledge bases.
*   **Modular Design:** Easily integrate with different LLMs, embedding models, and rerankers for customization.
*   **Comprehensive Deletion Capabilities:** Delete documents, entities, and relationships with optimized and data-consistent processes.
*   **Entity Merging:** Merge entities and their relationships.
*   **Token Usage Tracking:** Track token consumption by large language models.

---

## What's New

*   **[2025.06.16]** 🎯📢 Released [RAG-Anything](https://github.com/HKUDS/RAG-Anything), an All-in-One Multimodal RAG System.
*   **[2025.06.05]** 🎯📢 Added multimodal data handling through [RAG-Anything](https://github.com/HKUDS/RAG-Anything) integration.
*   **[2025.03.18]** 🎯📢 Added citation functionality.
*   **[2025.02.05]** 🎯📢 Released [VideoRAG](https://github.com/HKUDS/VideoRAG) understanding extremely long-context videos.
*   **[2025.01.13]** 🎯📢 Released [MiniRAG](https://github.com/HKUDS/MiniRAG) making RAG simpler with small models.
*   **[2025.01.06]** 🎯📢 Added PostgreSQL for Storage support.
*   **[2024.12.31]** 🎯📢 Added deletion by document ID support.
*   **[2024.11.25]** 🎯📢 Added custom knowledge graphs integration support.
*   **[2024.11.19]** 🎯📢 Comprehensive guide to LightRAG is now available on [LearnOpenCV](https://learnopencv.com/lightrag).
*   **[2024.11.11]** 🎯📢 Added deleting entities by their names support.
*   **[2024.11.09]** 🎯📢 Introduced the [LightRAG Gui](https://lightrag-gui.streamlit.app).
*   **[2024.11.04]** 🎯📢 Added Neo4J for Storage support.
*   **[2024.10.29]** 🎯📢 Added multiple file types support (PDF, DOC, PPT, CSV) via `textract`.
*   **[2024.10.20]** 🎯📢 Added Graph Visualization feature.
*   **[2024.10.18]** 🎯📢 Added link to a [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE).
*   **[2024.10.17]** 🎯📢 Created a [Discord channel](https://discord.gg/yF2MmDJyGJ)!
*   **[2024.10.16]** 🎯📢 Added Ollama models support.
*   **[2024.10.15]** 🎯📢 Added Hugging Face models support.

---

## Algorithm Flowchart

<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    Algorithm Flowchart
  </summary>

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*Figure 1: LightRAG Indexing Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*
![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*Figure 2: LightRAG Retrieval and Querying Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*

</details>

---

## Installation

### 1. Install LightRAG Server (Web UI and API)

Provides Web UI for document indexing, knowledge graph exploration, and RAG queries. Supports Ollama compatible interfaces.

*   **PyPI (Recommended):**
    ```bash
    pip install "lightrag-hku[api]"
    cp env.example .env
    lightrag-server
    ```
*   **From Source:**
    ```bash
    git clone https://github.com/HKUDS/LightRAG.git
    cd LightRAG
    # create a Python virtual enviroment if neccesary
    # Install in editable mode with API support
    pip install -e ".[api]"
    cp env.example .env
    lightrag-server
    ```
*   **Docker Compose:**
    ```bash
    git clone https://github.com/HKUDS/LightRAG.git
    cd LightRAG
    cp env.example .env
    # modify LLM and Embedding settings in .env
    docker compose up
    ```
    > Historical versions of LightRAG docker images can be found here: [LightRAG Docker Images]( https://github.com/HKUDS/LightRAG/pkgs/container/lightrag)

### 2. Install LightRAG Core

*   **From Source (Recommended):**
    ```bash
    cd LightRAG
    pip install -e .
    ```
*   **PyPI:**
    ```bash
    pip install lightrag-hku
    ```

---

## Quick Start

### LLM and Technology Stack Requirements

LightRAG requires powerful LLMs and proper configuration of Embedding and Reranker models for optimal performance.

*   **LLM Selection:**
    *   Use an LLM with at least 32 billion parameters (64KB context length recommended).
    *   Avoid reasoning models during indexing.
    *   Use more capable models during querying.
*   **Embedding Model:**
    *   Choose a high-performance embedding model (e.g., `BAAI/bge-m3`, `text-embedding-3-large`).
    *   The embedding model *must* be determined *before* document indexing and remain consistent during querying.  Re-indexing is needed when changing models.
*   **Reranker Model Configuration:**
    *   Configuring a Reranker model can significantly enhance LightRAG's retrieval performance.
    *   When a Reranker model is enabled, it is recommended to set the "mix mode" as the default query mode.
    *   Consider using Reranker models (e.g., `BAAI/bge-reranker-v2-m3`).

### Quick Start for LightRAG Server

*   Refer to [LightRAG Server](./lightrag/api/README.md) for details.

### Quick Start for LightRAG Core

1.  **Set up your OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY="sk-...your_opeai_key..."
    ```

2.  **Download a sample document:**

    ```bash
    curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
    ```

3.  **Run the demo code:**

    ```bash
    cd LightRAG
    python examples/lightrag_openai_demo.py
    ```

    For streaming responses, see `examples/lightrag_openai_compatible_demo.py`.  Modify the example's LLM and embedding configurations as needed.

    **Important Notes:**
    *   Different test scripts may use different embedding models. Clear the data directory (`./dickens`) when switching models.
    *   Only `lightrag_openai_demo.py` and `lightrag_openai_compatible_demo.py` are officially supported sample codes.

---

## Programming with LightRAG Core

> ⚠️ **For integration into your projects, use the REST API provided by the LightRAG Server.** LightRAG Core is primarily for embedded applications or research.

### ⚠️ Important: Initialization Requirements

**LightRAG requires explicit initialization.** Call `await rag.initialize_storages()` and `await initialize_pipeline_status()` after creating a `LightRAG` instance to avoid errors.

### A Simple Program

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    return rag

async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        await rag.ainsert("Your text")

        # Perform hybrid search
        mode = "hybrid"
        print(
          await rag.aquery(
              "What are the top themes in this story?",
              param=QueryParam(mode=mode)
          )
        )

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

**Key points:**

*   Set the `OPENAI_API_KEY` environment variable.
*   This example uses default storage settings in `WORKING_DIR/rag_storage`.
*   Shows the basic initialization: inject embedding and LLM functions, then initialize storage and pipeline status.

### LightRAG Initialization Parameters

<details>
<summary> Initialization Parameters </summary>

| **Parameter** | **Type** | **Explanation** | **Default** |
|--------------|----------|-----------------|-------------|
| **working_dir** | `str` | Directory where the cache will be stored | `lightrag_cache+timestamp` |
| **workspace** | str | Workspace name for data isolation between different LightRAG Instances |  |
| **kv_storage** | `str` | Storage type for documents and text chunks. Supported types: `JsonKVStorage`,`PGKVStorage`,`RedisKVStorage`,`MongoKVStorage` | `JsonKVStorage` |
| **vector_storage** | `str` | Storage type for embedding vectors. Supported types: `NanoVectorDBStorage`,`PGVectorStorage`,`MilvusVectorDBStorage`,`ChromaVectorDBStorage`,`FaissVectorDBStorage`,`MongoVectorDBStorage`,`QdrantVectorDBStorage` | `NanoVectorDBStorage` |
| **graph_storage** | `str` | Storage type for graph edges and nodes. Supported types: `NetworkXStorage`,`Neo4JStorage`,`PGGraphStorage`,`AGEStorage` | `NetworkXStorage` |
| **doc_status_storage** | `str` | Storage type for documents process status. Supported types: `JsonDocStatusStorage`,`PGDocStatusStorage`,`MongoDocStatusStorage` | `JsonDocStatusStorage` |
| **chunk_token_size** | `int` | Maximum token size per chunk when splitting documents | `1200` |
| **chunk_overlap_token_size** | `int` | Overlap token size between two chunks when splitting documents | `100` |
| **tokenizer** | `Tokenizer` | The function used to convert text into tokens (numbers) and back using .encode() and .decode() functions following `TokenizerInterface` protocol. If you don't specify one, it will use the default Tiktoken tokenizer. | `TiktokenTokenizer` |
| **tiktoken_model_name** | `str` | If you're using the default Tiktoken tokenizer, this is the name of the specific Tiktoken model to use. This setting is ignored if you provide your own tokenizer. | `gpt-4o-mini` |
| **entity_extract_max_gleaning** | `int` | Number of loops in the entity extraction process, appending history messages | `1` |
| **node_embedding_algorithm** | `str` | Algorithm for node embedding (currently not used) | `node2vec` |
| **node2vec_params** | `dict` | Parameters for node embedding | `{"dimensions": 1536,"num_walks": 10,"walk_length": 40,"window_size": 2,"iterations": 3,"random_seed": 3,}` |
| **embedding_func** | `EmbeddingFunc` | Function to generate embedding vectors from text | `openai_embed` |
| **embedding_batch_num** | `int` | Maximum batch size for embedding processes (multiple texts sent per batch) | `32` |
| **embedding_func_max_async** | `int` | Maximum number of concurrent asynchronous embedding processes | `16` |
| **llm_model_func** | `callable` | Function for LLM generation | `gpt_4o_mini_complete` |
| **llm_model_name** | `str` | LLM model name for generation | `meta-llama/Llama-3.2-1B-Instruct` |
| **summary_context_size** | `int` | Maximum tokens send to LLM to generate summaries for entity relation merging | `10000`（configured by env var SUMMARY_CONTEXT_SIZE) |
| **summary_max_tokens** | `int` | Maximum token size for entity/relation description | `500`（configured by env var SUMMARY_MAX_TOKENS) |
| **llm_model_max_async** | `int` | Maximum number of concurrent asynchronous LLM processes | `4`（default value changed by env var MAX_ASYNC) |
| **llm_model_kwargs** | `dict` | Additional parameters for LLM generation | |
| **vector_db_storage_cls_kwargs** | `dict` | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval | cosine_better_than_threshold: 0.2（default value changed by env var COSINE_THRESHOLD) |
| **enable_llm_cache** | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application | `TRUE` |
| **addon_params** | `dict` | Additional parameters, e.g., `{"language": "Simplified Chinese", "entity_types": ["organization", "person", "location", "event"]}`: sets example limit, entiy/relation extraction output language | language: English` |
| **embedding_cache_config** | `dict` | Configuration for question-answer caching. Contains three parameters: `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers. `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM. `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |

</details>

### Query Parameters (`QueryParam`)

Control query behavior using the `QueryParam` class:

```python
class QueryParam:
    # ... (See original README for full class definition)
```

### LLM and Embedding Injection

LightRAG requires you to provide your LLM and Embedding functions during initialization.

<details>
<summary> <b>Using Open AI-like APIs</b> </summary>

*   LightRAG supports OpenAI-compatible APIs:

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Using Hugging Face Models</b> </summary>

*   Example using Hugging Face Models:

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Using Ollama Models</b> </summary>

*   Example using Ollama Models:

```python
# (See original README for full example)
```

*   **Increasing context size**

    *   You can achieve this by:
        1.  **Increasing the `num_ctx` parameter in Modelfile**
        2.  **Setup `num_ctx` via Ollama API**
*   **Low RAM GPUs**

    *   In order to run this experiment on low RAM GPU you should select small model and tune context window (increasing context increase memory consumption).

</details>

<details>
<summary> <b>LlamaIndex</b> </summary>

LightRAG supports integration with LlamaIndex (`llm/llama_index_impl.py`):

-   Integrates with OpenAI and other providers through LlamaIndex
-   See [LlamaIndex Documentation](lightrag/llm/Readme.md) for detailed setup and examples

**Example Usage**

```python
# (See original README for full example)
```

</details>

### Rerank Function Injection

Enhance retrieval quality by injecting a reranking function (e.g., `cohere_rerank`, `jina_rerank`) into the `rerank_model_func` attribute of the `LightRAG` object.

### User Prompt vs. Query

Use `user_prompt` in `QueryParam` to guide the LLM on processing results *after* retrieval.  This separates search from output formatting.

```python
# (See original README for full example)
```

### Insert

<details>
  <summary> <b> Basic Insert </b></summary>

```python
# Basic Insert
rag.insert("Text")
```

</details>

<details>
  <summary> <b> Batch Insert </b></summary>

```python
# Basic Batch Insert: Insert multiple texts at once
rag.insert(["TEXT1", "TEXT2",...])

# Batch Insert with custom batch size configuration
rag = LightRAG(
    ...
    working_dir=WORKING_DIR,
    max_parallel_insert = 4
)

rag.insert(["TEXT1", "TEXT2", "TEXT3", ...])  # Documents will be processed in batches of 4
```

The `max_parallel_insert` parameter determines the number of documents processed concurrently in the document indexing pipeline. If unspecified, the default value is **2**. We recommend keeping this setting **below 10**, as the performance bottleneck typically lies with the LLM (Large Language Model) processing.The `max_parallel_insert` parameter determines the number of documents processed concurrently in the document indexing pipeline. If unspecified, the default value is **2**. We recommend keeping this setting **below 10**, as the performance bottleneck typically lies with the LLM (Large Language Model) processing.

</details>

<details>
  <summary> <b> Insert with ID </b></summary>

If you want to provide your own IDs for your documents, number of documents and number of IDs must be the same.

```python
# Insert single text, and provide ID for it
rag.insert("TEXT1", ids=["ID_FOR_TEXT1"])

# Insert multiple texts, and provide IDs for them
rag.insert(["TEXT1", "TEXT2",...], ids=["ID_FOR_TEXT1", "ID_FOR_TEXT2"])
```

</details>

<details>
  <summary><b>Insert using Pipeline</b></summary>

The `apipeline_enqueue_documents` and `apipeline_process_enqueue_documents` functions allow you to perform incremental insertion of documents into the graph.

This is useful for scenarios where you want to process documents in the background while still allowing the main thread to continue executing.

And using a routine to process new documents.

```python
rag = LightRAG(..)

await rag.apipeline_enqueue_documents(input)
# Your routine in loop
await rag.apipeline_process_enqueue_documents(input)
```

</details>

<details>
  <summary><b>Insert Multi-file Type Support</b></summary>

The `textract` supports reading file types such as TXT, DOCX, PPTX, CSV, and PDF.

```python
import textract

file_path = 'TEXT.pdf'
text_content = textract.process(file_path)

rag.insert(text_content.decode('utf-8'))
```

</details>

<details>
  <summary><b>Citation Functionality</b></summary>

By providing file paths, the system ensures that sources can be traced back to their original documents.

```python
# Define documents and their file paths
documents = ["Document content 1", "Document content 2"]
file_paths = ["path/to/doc1.txt", "path/to/doc2.txt"]

# Insert documents with file paths
rag.insert(documents, file_paths=file_paths)
```

</details>

### Storage

LightRAG uses 4 types of storage:

*   **KV_STORAGE:** LLM response cache, text chunks, document information.
*   **VECTOR_STORAGE:** Entities vectors, relation vectors, chunks vectors.
*   **GRAPH_STORAGE:** Entity relation graph.
*   **DOC_STATUS_STORAGE:** Document indexing status.

See the original README for a detailed breakdown of storage implementations (JsonKVStorage, PGKVStorage, RedisKVStorage, MongoKVStorage, etc.).

<details>
<summary> <b>Using Neo4J Storage</b> </summary>

*   For production-level scenarios, leverage an enterprise solution for KG storage.
*   Running Neo4J in Docker is recommended for local testing.
*   See: https://hub.docker.com/_/neo4j

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Using PostgreSQL Storage</b> </summary>

For production level scenarios you will most likely want to leverage an enterprise solution. PostgreSQL can provide a one-stop solution for you as KV store, VectorDB (pgvector) and GraphDB (apache AGE). PostgreSQL version 16.6 or higher is supported.

* PostgreSQL is lightweight,the whole binary distribution including all necessary plugins can be zipped to 40MB: Ref to [Windows Release](https://github.com/ShanGor/apache-age-windows/releases/tag/PG17%2Fv1.5.0-rc0) as it is easy to install for Linux/Mac.
* If you prefer docker, please start with this image if you are a beginner to avoid hiccups (DO read the overview): https://hub.docker.com/r/shangor/postgres-for-rag
* How to start? Ref to: [examples/lightrag_zhipu_postgres_demo.py](https://github.com/HKUDS/LightRAG/blob/main/examples/lightrag_zhipu_postgres_demo.py)
* For high-performance graph database requirements, Neo4j is recommended as Apache AGE's performance is not as competitive.

</details>

<details>
<summary> <b>Using Faiss Storage</b> </summary>
Before using Faiss vector database, you must manually install `faiss-cpu` or `faiss-gpu`.

- Install the required dependencies:

```
pip install faiss-cpu
```

You can also install `faiss-gpu` if you have GPU support.

- Here we are using `sentence-transformers` but you can also use `OpenAIEmbedding` model with `3072` dimensions.

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Using Memgraph for Storage</b> </summary>

*   Memgraph is a high-performance, in-memory graph database compatible with the Neo4j Bolt protocol.
*   You can run Memgraph locally using Docker for easy testing:
*   See: https://memgraph.com/download

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Using MongoDB Storage</b> </summary>

MongoDB provides a one-stop storage solution for LightRAG. MongoDB offers native KV storage and vector storage. LightRAG uses MongoDB collections to implement a simple graph storage. MongoDB's official vector search functionality (`$vectorSearch`) currently requires their official cloud service MongoDB Atlas. This functionality cannot be used on self-hosted MongoDB Community/Enterprise versions.

</details>

<details>
<summary> <b>Using Redis Storage</b> </summary>

LightRAG supports using Redis as KV storage. When using Redis storage, attention should be paid to persistence configuration and memory usage configuration. The following is the recommended Redis configuration:

```
# (See original README for full example)
```

</details>

### Data Isolation Between LightRAG Instances

The `workspace` parameter isolates data. Workspace implementations vary by storage type (file-based, collection-based, relational, Neo4j, etc.). See the original README for details.

## Edit Entities and Relations

LightRAG provides comprehensive knowledge graph management.

<details>
  <summary> <b> Create Entities and Relations </b></summary>

```python
# (See original README for full example)
```

</details>

<details>
  <summary> <b> Edit Entities and Relations </b></summary>

```python
# (See original README for full example)
```

</details>

<details>
  <summary> <b> Insert Custom KG </b></summary>

```python
# (See original README for full example)
```

</details>

<details>
  <summary> <b>Other Entity and Relation Operations</b></summary>

-   `create_entity`: Creates a new entity.
-   `edit_entity`: Updates or renames an existing entity.
-   `create_relation`: Creates a new relation between entities.
-   `edit_relation`: Updates an existing relation.

These operations maintain data consistency.

</details>

## Delete Functions

LightRAG provides document, entity, and relationship deletion.

<details>
<summary> <b>Delete Entities</b> </summary>

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Delete Relations</b> </summary>

```python
# (See original README for full example)
```

</details>

<details>
<summary> <b>Delete by Document ID</b> </summary>

```python
# (See original README for full example)
```

</details>

**Important Reminders:**

1.  Deletions are irreversible.
2.  Deleting large amounts of data may take time.
3.  Deletion maintains consistency between the graph and vector database.
4.  Backup your data before deletions.

**Batch Deletion Recommendations:**
- For batch deletion operations, consider using asynchronous methods for better performance
- For large-scale deletions, consider processing in batches to avoid excessive system load

## Entity Merging

<details>
<summary> <b>Merge Entities and Their Relationships</b> </summary>

```python
# (See original README for full example)
```

</details>

## Multimodal Document Processing (RAG-Anything Integration)

Leverage the power of [RAG-Anything](https://github.com/HKUDS/RAG-Anything) for advanced multimodal RAG:

**Key Features:**
- End-to-End Multimodal Pipeline
- Universal Document Support
- Specialized Content Analysis
- Multimodal Knowledge Graph
- Hybrid Intelligent Retrieval

**Quick Start:**

1.  Install RAG-Anything:

    ```bash
    pip install raganything
    ```

2.  Process multimodal documents:

    <details>
    <summary> <b> RAGAnything Usage Example </b></summary>

    ```python
        # (See original README for full example)
    ```
    </details>

For detailed documentation, please refer to the [RAG-Anything repository](https://github.com/HKUDS/RAG-Anything).

## Token Usage Tracking

<details>
<summary> <b>Overview and Usage</b> </summary>

Track token usage with the `TokenTracker` tool:

```python
# (See original README for full example)
```

### Usage Tips
- Use context managers for long sessions or batch operations to automatically track all token consumption
- For scenarios requiring segmented statistics, use manual mode and call reset() when appropriate
- Regular checking of token usage helps detect abnormal consumption early
- Actively use this feature during development and testing to optimize production costs

### Practical Examples
You can refer to these examples for implementing token tracking:
- `examples/lightrag_gemini_track_token_demo.py`: Token tracking example using Google Gemini model
- `examples/lightrag_siliconcloud_track_token_demo.py`: Token tracking example using SiliconCloud model

These examples demonstrate how to effectively use the TokenTracker feature with different models and scenarios.

</details>

## Data Export Functions

### Overview

Export your knowledge graph data.

### Export Functions

<details>
  <summary> <b> Basic Usage </b></summary>

```python
# (See original README for full example)
```

</details>

<details>
  <summary> <b> Different File Formats supported </b></summary>

```python
# (See original README for full example)
```
</details>

<details>
  <summary> <b> Additional Options </b></summary>

Include vector embeddings in the export (optional):

```python
# (See original README for full example)
```
</details>

### Data Included in Export

*   Entity information.
*   Relation data.
*   Relationship information from vector database.

## Cache

<details>
  <summary> <b>Clear Cache</b> </summary>

```python
# (See original README for full example)
```

Valid modes: "default", "naive", "local", "global", "hybrid", "mix"

</details>

## Troubleshooting

### Common Initialization Errors