<div align="center">

<div style="margin: 20px 0;">
  <img src="./assets/logo.png" width="120" height="120" alt="LightRAG Logo" style="border-radius: 20px; box-shadow: 0 8px 32px rgba(0, 217, 255, 0.3);">
</div>

# LightRAG: Unlock the Power of Advanced Retrieval-Augmented Generation 🚀

LightRAG is a simple yet powerful framework for Retrieval-Augmented Generation (RAG), enabling faster and more efficient access to information. **[Explore the original LightRAG repository here](https://github.com/HKUDS/LightRAG)**.

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

*   **Fast & Efficient RAG**: Optimized for speed and performance in Retrieval-Augmented Generation.
*   **Flexible Storage Options**: Supports various storage backends for vectors, graphs, and more, including:
    *   JsonKVStorage
    *   PGKVStorage      Postgres
    *   RedisKVStorage   Redis
    *   MongoKVStorage   MongoDB
    *   NetworkXStorage      NetworkX (default)
    *   Neo4JStorage         Neo4J
    *   PGGraphStorage       PostgreSQL with AGE plugin
    *   MemgraphStorage.     Memgraph
    *   NanoVectorDBStorage         NanoVector (default)
    *   PGVectorStorage             Postgres
    *   MilvusVectorDBStorage       Milvus
    *   FaissVectorDBStorage        Faiss
    *   QdrantVectorDBStorage       Qdrant
    *   MongoVectorDBStorage        MongoDB
    *   JsonDocStatusStorage        JsonFile (default)
    *   PGDocStatusStorage          Postgres
    *   MongoDocStatusStorage       MongoDB
*   **Advanced Knowledge Graph Capabilities:** Create, edit, and delete entities and relationships, with full support for data consistency and merging entities
*   **Multimodal Document Processing**: Seamlessly integrates with [RAG-Anything](https://github.com/HKUDS/RAG-Anything) for processing text, images, tables, and formulas.
*   **Comprehensive Data Export**: Export your knowledge graph data in CSV, Excel, Markdown, and text formats.
*   **Token Usage Tracking**: Monitor and manage LLM token consumption.
*   **Integration with Hugging Face and Ollama:** Utilize models hosted on Hugging Face and Ollama, in addition to OpenAI models.
*   **Flexible Query Options**: Supports local, global, hybrid, and mix query modes.

---

## News & Updates

*   [X] [2025.06.16]🎯📢Our team has released [RAG-Anything](https://github.com/HKUDS/RAG-Anything) an All-in-One Multimodal RAG System for seamless text, image, table, and equation processing.
*   [X] [2025.06.05]🎯📢LightRAG now supports comprehensive multimodal data handling through [RAG-Anything](https://github.com/HKUDS/RAG-Anything) integration, enabling seamless document parsing and RAG capabilities across diverse formats including PDFs, images, Office documents, tables, and formulas. Please refer to the new [multimodal section](https://github.com/HKUDS/LightRAG/?tab=readme-ov-file#multimodal-document-processing-rag-anything-integration) for details.
*   [X] [2025.03.18]🎯📢LightRAG now supports citation functionality, enabling proper source attribution.
*   [X] [2025.02.05]🎯📢Our team has released [VideoRAG](https://github.com/HKUDS/VideoRAG) understanding extremely long-context videos.
*   [X] [2025.01.13]🎯📢Our team has released [MiniRAG](https://github.com/HKUDS/MiniRAG) making RAG simpler with small models.
*   [X] [2025.01.06]🎯📢You can now [use PostgreSQL for Storage](#using-postgresql-for-storage).
*   [X] [2024.12.31]🎯📢LightRAG now supports [deletion by document ID](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete).
*   [X] [2024.11.25]🎯📢LightRAG now supports seamless integration of [custom knowledge graphs](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#insert-custom-kg), empowering users to enhance the system with their own domain expertise.
*   [X] [2024.11.19]🎯📢A comprehensive guide to LightRAG is now available on [LearnOpenCV](https://learnopencv.com/lightrag). Many thanks to the blog author.
*   [X] [2024.11.11]🎯📢LightRAG now supports [deleting entities by their names](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#delete).
*   [X] [2024.11.09]🎯📢Introducing the [LightRAG Gui](https://lightrag-gui.streamlit.app), which allows you to insert, query, visualize, and download LightRAG knowledge.
*   [X] [2024.11.04]🎯📢You can now [use Neo4J for Storage](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#using-neo4j-for-storage).
*   [X] [2024.10.29]🎯📢LightRAG now supports multiple file types, including PDF, DOC, PPT, and CSV via `textract`.
*   [X] [2024.10.20]🎯📢We've added a new feature to LightRAG: Graph Visualization.
*   [X] [2024.10.18]🎯📢We've added a link to a [LightRAG Introduction Video](https://youtu.be/oageL-1I0GE). Thanks to the author!
*   [X] [2024.10.17]🎯📢We have created a [Discord channel](https://discord.gg/yF2MmDJyGJ)! Welcome to join for sharing and discussions! 🎉🎉
*   [X] [2024.10.16]🎯📢LightRAG now supports [Ollama models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!
*   [X] [2024.10.15]🎯📢LightRAG now supports [Hugging Face models](https://github.com/HKUDS/LightRAG?tab=readme-ov-file#quick-start)!

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

The LightRAG Server is designed to provide Web UI and API support. The Web UI facilitates document indexing, knowledge graph exploration, and a simple RAG query interface. LightRAG Server also provide an Ollama compatible interfaces, aiming to emulate LightRAG as an Ollama chat model. This allows AI chat bot, such as Open WebUI, to access LightRAG easily.

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

LightRAG's demands on the capabilities of Large Language Models (LLMs) are significantly higher than those of traditional RAG, as it requires the LLM to perform entity-relationship extraction tasks from documents. Configuring appropriate Embedding and Reranker models is also crucial for improving query performance.

- **LLM Selection**:
  - It is recommended to use an LLM with at least 32 billion parameters.
  - The context length should be at least 32KB, with 64KB being recommended.
  - It is not recommended to choose reasoning models during the document indexing stage.
  - During the query stage, it is recommended to choose models with stronger capabilities than those used in the indexing stage to achieve better query results.
- **Embedding Model**:
  - A high-performance Embedding model is essential for RAG.
  - We recommend using mainstream multilingual Embedding models, such as: `BAAI/bge-m3` and `text-embedding-3-large`.
  - **Important Note**: The Embedding model must be determined before document indexing, and the same model must be used during the document query phase. For certain storage solutions (e.g., PostgreSQL), the vector dimension must be defined upon initial table creation. Therefore, when changing embedding models, it is necessary to delete the existing vector-related tables and allow LightRAG to recreate them with the new dimensions.
- **Reranker Model Configuration**:
  - Configuring a Reranker model can significantly enhance LightRAG's retrieval performance.
  - When a Reranker model is enabled, it is recommended to set the "mix mode" as the default query mode.
  - We recommend using mainstream Reranker models, such as: `BAAI/bge-reranker-v2-m3` or models provided by services like Jina.

### Quick Start for LightRAG Server

* For more information about LightRAG Server, please refer to [LightRAG Server](./lightrag/api/README.md).

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

> ⚠️ **If you would like to integrate LightRAG into your project, we recommend utilizing the REST API provided by the LightRAG Server**. LightRAG Core is typically intended for embedded applications or for researchers who wish to conduct studies and evaluations.

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

Important notes for the above snippet:

- Export your OPENAI_API_KEY environment variable before running the script.
- This program uses the default storage settings for LightRAG, so all data will be persisted to WORKING_DIR/rag_storage.
- This program demonstrates only the simplest way to initialize a LightRAG object: Injecting the embedding and LLM functions, and initializing storage and pipeline status after creating the LightRAG object.

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

> default value of Top_k can be change by environment  variables  TOP_K.

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

* If you want to use Hugging Face models, you only need to set LightRAG as follows:

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
        func=lambda texts: ollama_embed(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

* **Increasing context size**

In order for LightRAG to work context should be at least 32k tokens. By default Ollama models have context size of 8k. You can achieve this using one of two ways:

* **Increasing the `num_ctx` parameter in Modelfile**

1. Pull the model:

```bash
ollama pull qwen2
```

2. Display the model file:

```bash
ollama show --modelfile qwen2 > Modelfile
```

3. Edit the Modelfile by adding the following line:

```bash
PARAMETER num_ctx 32768
```

4. Create the modified model:

```bash
ollama create -f Modelfile qwen2m
```

* **Setup `num_ctx` via Ollama API**

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
        func=lambda texts: ollama_embed(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)
```

* **Low RAM GPUs**

In order to run this experiment on low RAM GPU you should select small model and tune context window (increasing context increase memory consumption). For example, running this ollama example on repurposed mining GPU with 6Gb of RAM required to set context size to 26k while using `gemma2:2b`. It was able to find 197 entities