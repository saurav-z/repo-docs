<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# LightRAG: Revolutionizing Retrieval-Augmented Generation (RAG)

**LightRAG is a powerful and fast framework that simplifies Retrieval-Augmented Generation, enabling you to build advanced, knowledge-rich applications.**

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

## Key Features of LightRAG

*   **Simplified RAG Implementation:** Easily integrate and build RAG applications.
*   **Fast Performance:** Optimized for speed, allowing efficient data retrieval and generation.
*   **Flexible Storage Options:** Support for various storage backends, including:
    *   JSON
    *   PostgreSQL
    *   Redis
    *   MongoDB
    *   Neo4j
    *   Milvus
    *   Faiss
    *   Qdrant
    *   Memgraph
*   **Multimodal Capabilities:** Integrates seamlessly with [RAG-Anything](https://github.com/HKUDS/RAG-Anything) for processing text, images, tables, and more.
*   **Knowledge Graph Integration:** Create, edit, and delete entities and relations within your knowledge graph.
*   **Flexible Querying:** Supports local, global, and hybrid search modes, and customizable query parameters.
*   **Citation & Source Attribution:** Track and attribute sources with citation functionality.
*   **Token Usage Tracking:** Monitor and manage token consumption for LLMs.
*   **Data Export:** Export your knowledge graph in various formats.
*   **Easy Integration:** Supports LlamaIndex.
*   **Graph Visualization:** Visualise your knowledge graph with the LightRAG server.

## What's New

*   **[2025.06.16]** 🎯📢 Released [RAG-Anything](https://github.com/HKUDS/RAG-Anything): All-in-One Multimodal RAG System.
*   **[2025.06.05]** 🎯📢  Multimodal data handling through [RAG-Anything](https://github.com/HKUDS/RAG-Anything) integration.
*   **[2025.03.18]** 🎯📢  Added citation functionality.
*   **[2025.02.05]** 🎯📢  Released [VideoRAG](https://github.com/HKUDS/VideoRAG).
*   **[2025.01.13]** 🎯📢  Released [MiniRAG](https://github.com/HKUDS/MiniRAG).
*   **[2025.01.06]** 🎯📢  PostgreSQL for Storage.
*   **[2024.12.31]** 🎯📢  Deletion by document ID.
*   **[2024.11.25]** 🎯📢  Custom knowledge graphs integration.
*   **[2024.11.19]** 🎯📢  A comprehensive guide on [LearnOpenCV](https://learnopencv.com/lightrag).
*   **[2024.11.11]** 🎯📢  Deleting entities by their names.
*   **[2024.11.09]** 🎯📢  LightRAG Gui is released: [LightRAG Gui](https://lightrag-gui.streamlit.app).
*   **[2024.11.04]** 🎯📢  Neo4J for Storage.
*   **[2024.10.29]** 🎯📢  Multiple file types supported via `textract`.
*   **[2024.10.20]** 🎯📢  Graph Visualization is released.
*   **[2024.10.18]** 🎯📢  LightRAG Introduction Video: [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE).
*   **[2024.10.17]** 🎯📢  Discord channel: [Discord channel](https://discord.gg/yF2MmDJyGJ).
*   **[2024.10.16]** 🎯📢  Ollama models are supported.
*   **[2024.10.15]** 🎯📢  Hugging Face models are supported.

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

### Install LightRAG Server

The LightRAG Server provides Web UI and API support for easy document indexing, knowledge graph exploration, and RAG queries. It also provides an Ollama-compatible interface.

*   **Install from PyPI:**

    ```bash
    pip install "lightrag-hku[api]"
    cp env.example .env
    lightrag-server
    ```

*   **Install from Source:**

    ```bash
    git clone https://github.com/HKUDS/LightRAG.git
    cd LightRAG
    # create a Python virtual enviroment if neccesary
    # Install in editable mode with API support
    pip install -e ".[api]"
    cp env.example .env
    lightrag-server
    ```

*   **Launch with Docker Compose:**

    ```bash
    git clone https://github.com/HKUDS/LightRAG.git
    cd LightRAG
    cp env.example .env
    # modify LLM and Embedding settings in .env
    docker compose up
    ```

    >   Historical versions of LightRAG docker images can be found here: [LightRAG Docker Images]( https://github.com/HKUDS/LightRAG/pkgs/container/lightrag)

### Install LightRAG Core

*   **Install from Source (Recommended):**

    ```bash
    cd LightRAG
    pip install -e .
    ```

*   **Install from PyPI:**

    ```bash
    pip install lightrag-hku
    ```

---

## Quick Start

### Requirements

LightRAG requires a powerful LLM and proper embedding & reranker models.

*   **LLM:** Recommended with at least 32 billion parameters and a context length of at least 32KB.
*   **Embedding Model:** Use a high-performance multilingual embedding model, like `BAAI/bge-m3` or `text-embedding-3-large`.  **Important:**  Use the same embedding model during indexing and querying, and redefine vector dimensions if you change embedding models.
*   **Reranker Model:** Configuring a reranker can significantly improve retrieval performance. Recommended models: `BAAI/bge-reranker-v2-m3` or Jina.

### Quick Start for LightRAG Server

Refer to [LightRAG Server](./lightrag/api/README.md) for more information.

### Quick Start for LightRAG Core

Follow the steps below or see the [video demo](https://www.youtube.com/watch?v=g21royNJ4fw) to guide you through the local setup process.

If you have an OpenAI API key, you can run the demo:

```bash
cd LightRAG
export OPENAI_API_KEY="sk-...your_opeai_key..."
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt
python examples/lightrag_openai_demo.py
```

For streaming response example, please see `examples/lightrag_openai_compatible_demo.py`. Before execution, ensure you modify the sample code's LLM and embedding configurations accordingly.

**Note 1**: Remember to clear the data directory (`./dickens`) if you switch to a different embedding model, or the program may encounter errors.
**Note 2**: Only `lightrag_openai_demo.py` and `lightrag_openai_compatible_demo.py` are officially supported examples.

---

## Programming with LightRAG Core

>  **Integrate LightRAG into your project using the REST API provided by the LightRAG Server.** LightRAG Core is for embedded applications or research.

### Initialization

**LightRAG requires explicit initialization before use.** Call both `await rag.initialize_storages()` and `await initialize_pipeline_status()` after creating a LightRAG instance.

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

Important notes:

*   Set your `OPENAI_API_KEY` environment variable.
*   Data is persisted to `WORKING_DIR/rag_storage`.
*   This demonstrates basic initialization: Injecting embedding and LLM functions, initializing storage and pipeline status.

### LightRAG init parameters

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
| **summary_max_tokens** | `int` | Maximum tokens send to LLM to generate entity relation summaries | `30000`（configured by env var SUMMARY_MAX_TOKENS) |
| **llm_model_max_async** | `int` | Maximum number of concurrent asynchronous LLM processes | `4`（default value changed by env var MAX_ASYNC) |
| **llm_model_kwargs** | `dict` | Additional parameters for LLM generation | |
| **vector_db_storage_cls_kwargs** | `dict` | Additional parameters for vector database, like setting the threshold for nodes and relations retrieval | cosine_better_than_threshold: 0.2（default value changed by env var COSINE_THRESHOLD) |
| **enable_llm_cache** | `bool` | If `TRUE`, stores LLM results in cache; repeated prompts return cached responses | `TRUE` |
| **enable_llm_cache_for_entity_extract** | `bool` | If `TRUE`, stores LLM results in cache for entity extraction; Good for beginners to debug your application | `TRUE` |
| **addon_params** | `dict` | Additional parameters, e.g., `{"example_number": 1, "language": "Simplified Chinese", "entity_types": ["organization", "person", "geo", "event"]}`: sets example limit, entiy/relation extraction output language | `example_number: all examples, language: English` |
| **embedding_cache_config** | `dict` | Configuration for question-answer caching. Contains three parameters: `enabled`: Boolean value to enable/disable cache lookup functionality. When enabled, the system will check cached responses before generating new answers. `similarity_threshold`: Float value (0-1), similarity threshold. When a new question's similarity with a cached question exceeds this threshold, the cached answer will be returned directly without calling the LLM. `use_llm_check`: Boolean value to enable/disable LLM similarity verification. When enabled, LLM will be used as a secondary check to verify the similarity between questions before returning cached answers. | Default: `{"enabled": False, "similarity_threshold": 0.95, "use_llm_check": False}` |

</details>

### Query Param

Use `QueryParam` to configure your query:

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

    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", "20"))
    """Number of text chunks to retrieve initially from vector search and keep after reranking.
    If None, defaults to top_k value.
    """

    max_entity_tokens: int = int(os.getenv("MAX_ENTITY_TOKENS", "6000"))
    """Maximum number of tokens allocated for entity context in unified token control system."""

    max_relation_tokens: int = int(os.getenv("MAX_RELATION_TOKENS", "8000"))
    """Maximum number of tokens allocated for relationship context in unified token control system."""

    max_total_tokens: int = int(os.getenv("MAX_TOTAL_TOKENS", "30000"))
    """Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt)."""

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Stores past conversation history to maintain context.
    Format: [{"role": "user/assistant", "content": "message"}].
    """

    # Deprated: history message have negtive effect on query performance
    history_turns: int = 0
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

>  The `TOP_K` environment variable can change the default `top_k` value.

### LLM and Embedding Injection

LightRAG requires LLM and Embedding models. During initialization, inject model invocation methods:

<details>
<summary> <b>Using Open AI-like APIs</b> </summary>

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

Install your model and embedding model (e.g., `nomic-embed-text`).

```python
# Initialize LightRAG with Ollama model
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='your_model_name', # Your model name
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        func=lambda texts: ollama_embed(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

*   **Increasing context size:**

    *   **Modifying Modelfile:**  Add `PARAMETER num_ctx 32768` to your model's `Modelfile` and create a new model.
    *   **Setup `num_ctx` via Ollama API** Use `llm_model_kwargs={"options": {"num_ctx": 32768}}`

*   **Low RAM GPUs**
    *   Select a smaller model and tune the context window.
</details>
<details>
<summary> <b>LlamaIndex</b> </summary>

LightRAG supports LlamaIndex integration:

-   Integrates with OpenAI and other providers.

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
</details>

### Rerank Function Injection

Enhance retrieval quality with reranking using these providers:
*   Cohere / vLLM: `cohere_rerank`
*   Jina AI: `jina_rerank`
*   Aliyun: `ali_rerank`

Inject one into the `rerank_model_func` attribute of the LightRAG object.

### User Prompt vs. Query

Use `user_prompt` in `QueryParam` to guide the LLM in post-retrieval processing, without affecting the RAG search:

```python
# Create query parameters
query_param = QueryParam(
    mode = "hybrid",  # Other modes：local, global, hybrid, mix, naive
    user_prompt = "For diagrams, use mermaid format with English/Pinyin node names and Chinese display labels",
)

# Query and process
response_default = rag.query(
    "Please draw a character relationship diagram for Scrooge",
    param=query_param
)
print(response_default)
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