<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# 🚀 LightRAG: Simple, Fast, and Powerful Retrieval-Augmented Generation (RAG)

LightRAG is a cutting-edge open-source framework that empowers you to build Retrieval-Augmented Generation (RAG) systems with ease and speed, providing enhanced performance and advanced features.  [Explore the original repository](https://github.com/HKUDS/LightRAG)

<div align="center">
    <a href="https://trendshift.io/repositories/13043" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13043" alt="HKUDS%2FLightRAG | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

<div align="center">
  <div style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

<div align="center">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; text-align: center;">
    <p>
      <a href='https://github.com/HKUDS/LightRAG'><img src='https://img.shields.io/badge/🔥Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'></a>
      <a href='https://arxiv.org/abs/2410.05779'><img src='https://img.shields.io/badge/📄arXiv-2410.05779-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'></a>
      <a href="https://github.com/HKUDS/LightRAG/stargazers"><img src='https://img.shields.io/github/stars/HKUDS/LightRAG?color=00d9ff&style=for-the-badge&logo=star&logoColor=white&labelColor=1a1a2e' /></a>
    </p>
    <p>
      <img src="https://img.shields.io/badge/🐍Python-3.10-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
      <a href="https://pypi.org/project/lightrag-hku/"><img src="https://img.shields.io/pypi/v/lightrag-hku.svg?style=for-the-badge&logo=pypi&logoColor=white&labelColor=1a1a2e&color=ff6b6b"></a>
    </p>
    <p>
      <a href="https://discord.gg/yF2MmDJyGJ"><img src="https://img.shields.io/badge/💬Discord-Community-7289da?style=for-the-badge&logo=discord&logoColor=white&labelColor=1a1a2e"></a>
      <a href="https://github.com/HKUDS/LightRAG/issues/285"><img src="https://img.shields.io/badge/💬WeChat-Group-07c160?style=for-the-badge&logo=wechat&logoColor=white&labelColor=1a1a2e"></a>
    </p>
    <p>
      <a href="README-zh.md"><img src="https://img.shields.io/badge/🇨🇳中文版-1a1a2e?style=for-the-badge"></a>
      <a href="README.md"><img src="https://img.shields.io/badge/🇺🇸English-1a1a2e?style=for-the-badge"></a>
    </p>
  </div>
</div>

</div>

<div align="center" style="margin: 30px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="800">
</div>

<div align="center" style="margin: 30px 0;">
    <img src="./README.assets/b2aaf634151b4706892693ffb43d9093.png" width="800" alt="LightRAG Diagram">
</div>

---

## Key Features

*   **Simplified RAG Implementation:**  Easily build and deploy RAG systems with minimal code.
*   **Fast Performance:** Optimized for speed, enabling efficient retrieval and generation.
*   **Multimodal Document Processing:**  Seamlessly handle various data formats with RAG-Anything integration.
*   **Flexible Storage Options:**  Supports multiple storage backends, including local JSON, PostgreSQL, Neo4j, and more.
*   **Advanced Knowledge Graph Features:** Create, edit, and delete entities and relationships for richer knowledge representation.
*   **Entity Merging:** Merge entities and their relationships for better data consistency.
*   **Comprehensive Deletion Capabilities:**  Delete documents, entities, and relationships efficiently.
*   **Token Usage Tracking:** Monitor and manage LLM token consumption for cost optimization.
*   **Data Export:** Export data in multiple formats for analysis and sharing.
*   **Conversation History Support:** Enable multi-turn dialogue for more engaging interactions.

---

## What's New

*   **[2025.06.16]** RAG-Anything for All-in-One Multimodal RAG System
*   **[2025.06.05]** Multimodal data handling through [RAG-Anything](https://github.com/HKUDS/RAG-Anything) integration
*   **[2025.03.18]** Citation functionality
*   **[2025.02.05]** VideoRAG understanding extremely long-context videos
*   **[2025.01.13]** MiniRAG making RAG simpler with small models
*   **[2025.01.06]** PostgreSQL Storage Support
*   **[2024.12.31]** Delete by document ID
*   **[2024.11.25]** Custom Knowledge Graphs Integration
*   **[2024.11.19]** Comprehensive Guide on LearnOpenCV
*   **[2024.11.11]** Delete entities by names
*   **[2024.11.09]** LightRAG GUI
*   **[2024.11.04]** Neo4J Storage Support
*   **[2024.10.29]** Multiple file types support via `textract`
*   **[2024.10.20]** Graph Visualization feature
*   **[2024.10.18]** LightRAG Introduction Video link
*   **[2024.10.17]** Discord channel
*   **[2024.10.16]** Ollama model support
*   **[2024.10.15]** Hugging Face model support

<details>
  <summary style="font-size: 1.4em; font-weight: bold; cursor: pointer; display: list-item;">
    Algorithm Flowchart
  </summary>

![LightRAG Indexing Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-VectorDB-Json-KV-Store-Indexing-Flowchart-scaled.jpg)
*Figure 1: LightRAG Indexing Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*
![LightRAG Retrieval and Querying Flowchart](https://learnopencv.com/wp-content/uploads/2024/11/LightRAG-Querying-Flowchart-Dual-Level-Retrieval-Generation-Knowledge-Graphs-scaled.jpg)
*Figure 2: LightRAG Retrieval and Querying Flowchart - Img Caption : [Source](https://learnopencv.com/lightrag/)*

</details>

## Installation

### Install LightRAG Server

The LightRAG Server provides Web UI and API support.

* Install from PyPI

```bash
pip install "lightrag-hku[api]"
cp env.example .env
lightrag-server
```

* Installation from Source

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
# create a Python virtual enviroment if neccesary
# Install in editable mode with API support
pip install -e ".[api]"
cp env.example .env
lightrag-server
```

* Launching the LightRAG Server with Docker Compose

```
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
cp env.example .env
# modify LLM and Embedding settings in .env
docker compose up
```

> Historical versions of LightRAG docker images can be found here: [LightRAG Docker Images]( https://github.com/HKUDS/LightRAG/pkgs/container/lightrag)

### Install  LightRAG Core

* Install from source (Recommend)

```bash
cd LightRAG
pip install -e .
```

* Install from PyPI

```bash
pip install lightrag-hku
```

## Quick Start

### LLM and Technology Stack Requirements for LightRAG

LightRAG requires a Large Language Model (LLM) to perform entity-relationship extraction tasks from documents. Configuring appropriate Embedding and Reranker models is also crucial for improving query performance.

-   **LLM Selection**:
    -   It is recommended to use an LLM with at least 32 billion parameters.
    -   The context length should be at least 32KB, with 64KB being recommended.
-   **Embedding Model**:
    -   A high-performance Embedding model is essential for RAG.
    -   We recommend using mainstream multilingual Embedding models, such as: `BAAI/bge-m3` and `text-embedding-3-large`.
    -   **Important Note**: The Embedding model must be determined before document indexing, and the same model must be used during the document query phase.
-   **Reranker Model Configuration**:
    -   Configuring a Reranker model can significantly enhance LightRAG's retrieval performance.
    -   When a Reranker model is enabled, it is recommended to set the "mix mode" as the default query mode.
    -   We recommend using mainstream Reranker models, such as: `BAAI/bge-reranker-v2-m3` or models provided by services like Jina.

### Quick Start for LightRAG Server

*   For more information about LightRAG Server, please refer to [LightRAG Server](./lightrag/api/README.md).

### Quick Start for LightRAG core

To get started with LightRAG core, refer to the sample codes available in the `examples` folder. Additionally, a [video demo](https://www.youtube.com/watch?v=g21royNJ4fw) demonstration is provided to guide you through the local setup process. If you already possess an OpenAI API key, you can run the demo right away:

```bash
### you should run the demo code with project folder
cd LightRAG
### provide your API-KEY for OpenAI
export OPENAI_API_KEY="sk-...your_opeai_key..."
### download the demo document of "A Christmas Carol" by Charles Dickens
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
### run the demo code
python examples/lightrag_openai_demo.py
```

For a streaming response implementation example, please see `examples/lightrag_openai_compatible_demo.py`. Prior to execution, ensure you modify the sample code's LLM and embedding configurations accordingly.

**Note 1**: When running the demo program, please be aware that different test scripts may use different embedding models. If you switch to a different embedding model, you must clear the data directory (`./dickens`); otherwise, the program may encounter errors. If you wish to retain the LLM cache, you can preserve the `kv_store_llm_response_cache.json` file while clearing the data directory.

**Note 2**: Only `lightrag_openai_demo.py` and `lightrag_openai_compatible_demo.py` are officially supported sample codes. Other sample files are community contributions that haven't undergone full testing and optimization.

## Programing with LightRAG Core

> If you would like to integrate LightRAG into your project, we recommend utilizing the REST API provided by the LightRAG Server. LightRAG Core is typically intended for embedded applications or for researchers who wish to conduct studies and evaluations.

### ⚠️ Important: Initialization Requirements

**LightRAG requires explicit initialization before use.** You must call both `await rag.initialize_storages()` and `await initialize_pipeline_status()` after creating a LightRAG instance, otherwise you will encounter errors like:
- `AttributeError: __aenter__` - if storages are not initialized
- `KeyError: 'history_messages'` - if pipeline status is not initialized

### A Simple Program

Use the below Python snippet to initialize LightRAG, insert text to it, and perform queries:

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
        rag.insert("Your text")

        # Perform hybrid search
        mode = "hybrid"
        print(
          await rag.query(
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

Important notes for the above snippet:

-   Export your OPENAI_API_KEY environment variable before running the script.
-   This program uses the default storage settings for LightRAG, so all data will be persisted to WORKING_DIR/rag_storage.
-   This program demonstrates only the simplest way to initialize a LightRAG object: Injecting the embedding and LLM functions, and initializing storage and pipeline status after creating the LightRAG object.

### LightRAG init parameters

A full list of LightRAG init parameters:

<details>
<summary> Parameters </summary>

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
| **llm_model_max_token_size** | `int` | Maximum tokens send to LLM to generate entity relation summaries | `32000`（default value changed by env var MAX_TOKENS) |
| **llm_model_max_async** | `int` | Maximum number of concurrent asynchronous LLM processes | `4`（default value changed by env var MAX_ASYNC) |
| **llm_model_kwargs** | `dict` | Additional parameters for LLM generation | |
| **vector_db_storage_cls_kwargs** | `dict` | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval | cosine_better_than_threshold: 0.2（default value changed by env var COSINE_THRESHOLD) |
| **enable_llm_cache** | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application | `TRUE` |
| **addon_params** | `dict` | Additional parameters, e.g., `{"example_number": 1, "language": "Simplified Chinese", "entity_types": ["organization", "person", "geo", "event"]}`: sets example limit, entiy/relation extraction output language | `example_number: all examples, language: English` |
| **convert_response_to_json_func** | `callable` | Not used | `convert_response_to_json` |
| **embedding_cache_config** | `dict` | Configuration for question-answer caching. Contains three parameters: `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers. `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM. `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |

</details>

### Query Param

Use QueryParam to control the behavior your query:

```python
class QueryParam:
    """Configuration parameters for query execution in LightRAG."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "global"
    """Specifies the retrieval mode:
    - "local": Focuses on context-dependent information.
    - "global": Utilizes global knowledge.
    - "hybrid": Combines local and global retrieval methods.
    - "naive": Performs a basic search without advanced techniques.
    - "mix": Integrates knowledge graph and vector retrieval.
    """

    only_need_context: bool = False
    """If True, only returns the retrieved context without generating a response."""

    only_need_prompt: bool = False
    """If True, only returns the generated prompt without producing a response."""

    response_type: str = "Multiple Paragraphs"
    """Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'."""

    stream: bool = False
    """If True, enables streaming output for real-time responses."""

    top_k: int = int(os.getenv("TOP_K", "60"))
    """Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode."""

    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", "10"))
    """Number of text chunks to retrieve initially from vector search and keep after reranking.
    If None, defaults to top_k value.
    """

    max_entity_tokens: int = int(os.getenv("MAX_ENTITY_TOKENS", "10000"))
    """Maximum number of tokens allocated for entity context in unified token control system."""

    max_relation_tokens: int = int(os.getenv("MAX_RELATION_TOKENS", "10000"))
    """Maximum number of tokens allocated for relationship context in unified token control system."""

    max_total_tokens: int = int(os.getenv("MAX_TOTAL_TOKENS", "32000"))
    """Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt)."""

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    history_turns: int = 3
    """Number of complete conversation turns (user-assistant pairs) to consider in the response context."""

    ids: list[str] | None = None
    """List of ids to filter the results."""

    model_func: Callable[..., object] | None = None
    """Optional override for the LLM model function to use for this specific query.
    If provided, this will be used instead of the global model function.
    This allows using different models for different query modes.
    """

    user_prompt: str | None = None
    """User-provided prompt for the query.
    If proivded, this will be use instead of the default vaulue from prompt template.
    """

    enable_rerank: bool = True
    """Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued.
    Default is True to enable reranking when rerank model is available.
    """
```

> default value of Top\_k can be change by environment  variables  TOP_K.

### LLM and Embedding Injection

LightRAG requires the utilization of LLM and Embedding models to accomplish document indexing and querying tasks. During the initialization phase, it is necessary to inject the invocation methods of the relevant models into LightRAG：

<details>
<summary> <b>Using Open AI-like APIs</b> </summary>

* LightRAG also supports Open AI-like chat/embeddings APIs:

```python
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model="solar-embedding-1-large-query",
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar"
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=4096,
            max_token_size=8192,
            func=embedding_func
        )
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag
```

</details>

<details>
<summary> <b>Using Hugging Face Models</b> </summary>

*   If you want to use Hugging Face models, you only need to set LightRAG as follows:

    See `lightrag_hf_demo.py`

```python
# Initialize LightRAG with Hugging Face model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # Use Hugging Face model for text generation
    llm_model_name='meta-llama/Llama-3.1-8B-Instruct',  # Model name from Hugging Face
    # Use Hugging Face embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            embed_model=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        )
    ),
)
```

</details>

<details>
<summary> <b>Using Ollama Models</b> </summary>
**Overview**

If you want to use Ollama models, you need to pull model you plan to use and embedding model, for example `nomic-embed-text`.

Then you only need to set LightRAG as follows:

```python
# Initialize LightRAG with Ollama model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

*   **Increasing context size**

    In order for LightRAG to work context should be at least 32k tokens. By default Ollama models have context size of 8k. You can achieve this using one of two ways:

    *   **Increasing the `num_ctx` parameter in Modelfile**

        1.  Pull the model:

            ```bash
            ollama pull qwen2
            ```

        2.  Display the model file:

            ```bash
            ollama show --modelfile qwen2 > Modelfile
            ```

        3.  Edit the Modelfile by adding the following line:

            ```bash
            PARAMETER num_ctx 32768
            ```

        4.  Create the modified model:

            ```bash
            ollama create -f Modelfile qwen2m
            ```

    *   **Setup `num_ctx` via Ollama API**

        Tiy can use `llm_model_kwargs` param to configure ollama:

        ```python
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
            llm_model_name='your_model_name', # Your model name
            llm_model_kwargs={"options": {"num_ctx": 32768}},
            # Use Ollama embedding function
            embedding_func=EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts,
                    embed_model="nomic-embed-text"
                )
            ),
        )
        ```

    *   **Low RAM GPUs**

        In order to run this experiment on low RAM GPU you should select small model and tune context window (increasing context increase memory consumption). For example, running this ollama example on repurposed mining GPU with 6Gb of RAM required to set context size to 26k while using `gemma2:2b`. It was able to find 197 entities and 19 relations on `book.txt`.

</details>
<details>
<summary> <b>LlamaIndex</b> </summary>

LightRAG supports integration with LlamaIndex (`llm/llama_index_impl.py`):

-   Integrates with OpenAI and other providers through LlamaIndex
-   See [LlamaIndex Documentation](lightrag/llm/Readme.md) for detailed setup and examples

**Example Usage**

```python
# Using LlamaIndex with direct OpenAI access
import asyncio
from lightrag import LightRAG
from lightrag.llm.llama_index_impl import llama_index_complete_if_cache, llama_index_embed
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup log handler for LightRAG
setup_logger("lightrag", level="INFO")

async def initialize_rag():
    rag = LightRAG(
        working_dir="your/path",
        llm_model_func=llama_index_complete_if_cache,  # LlamaIndex-compatible completion function
        embedding_func=EmbeddingFunc(    # LlamaIndex-compatible embedding function
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: llama_index_embed(texts, embed_model=embed_model)
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Perform naive search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
    )

    # Perform local search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
    )

    # Perform global search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
    )

    # Perform hybrid search
    print(
        rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
    )

if __name__ == "__main__":
    main()
```

**For detailed documentation and examples, see:**

-   [LlamaIndex Documentation](lightrag/llm/Readme.md)
-   [Direct OpenAI Example](examples/lightrag_llamaindex_direct_demo.py)
-   [LiteLLM Proxy Example](examples/lightrag_llamaindex_litellm_demo.py)

</details>

### Conversation History Support

LightRAG now supports multi-turn dialogue through the conversation history feature. Here's how to use it:

<details>
  <summary> <b> Usage Example </b></summary>

```python
# Create conversation history
conversation_history = [
    {"role": "user", "content": "What is the main character's attitude towards Christmas?"},
    {"role": "assistant", "content": "At the beginning of the story, Ebenezer Scrooge has a very negative attitude towards Christmas..."},
    {"role": "user", "content": "How does his attitude change?"}
]

# Create query parameters with conversation history
query_param = QueryParam(
    mode="mix",  # or any other mode: "local", "global", "hybrid"
    conversation_history=conversation_history,  # Add the conversation history
    history_turns=3  #